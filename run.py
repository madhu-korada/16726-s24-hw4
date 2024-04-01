import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torchvision.models as models
import copy
import sys
from utils import load_image, Normalization, device, imshow, get_image_optimizer
from style_and_content import ContentLoss, StyleLoss


"""A ``Sequential`` module contains an ordered list of child modules. For
instance, ``vgg19.features`` contains a sequence (Conv2d, ReLU, MaxPool2d,
Conv2d, ReLU…) aligned in the right order of depth. We need to add our
content loss and style loss layers immediately after the convolution
layer they are detecting. To do this we must create a new ``Sequential``
module that has content loss and style loss modules correctly inserted.
"""

# desired depth layers to compute style/content losses :
content_layers_default = ['conv_4']
style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']


def get_model_and_losses(cnn, style_img, content_img,
                               content_layers=content_layers_default,
                               style_layers=style_layers_default):
    cnn = copy.deepcopy(cnn)

    # build a sequential model consisting of a Normalization layer
    # then all the layers of the VGG feature network along with ContentLoss and StyleLoss
    # layers in the specified places

    # just in order to have an iterable access to or list of content/syle
    # losses
    content_losses = []
    style_losses = []

    # assuming that cnn is a nn.Sequential, so we make a new nn.Sequential
    # to put in modules that are supposed to be activated sequentially
    # here if you need a nn.ReLU layer, make sure to use inplace=False
    # as the in place version interferes with the loss layers
    # trim off the layers after the last content and style losses
    # as they are vestigial

    normalization = Normalization().to(device)
    model = nn.Sequential(normalization)

    
    i = 0  # increment every time we see a conv
    for layer in cnn.children():
        if isinstance(layer, nn.Conv2d):
            i += 1
            name = 'conv_{}'.format(i)
        elif isinstance(layer, nn.ReLU):
            name = 'relu_{}'.format(i)
            # The in-place version doesn't play very nicely with the ``ContentLoss``
            # and ``StyleLoss`` we insert below. So we replace with out-of-place
            # ones here.
            layer = nn.ReLU(inplace=False)
        elif isinstance(layer, nn.MaxPool2d):
            name = 'pool_{}'.format(i)
        elif isinstance(layer, nn.BatchNorm2d):
            name = 'bn_{}'.format(i)
        else:
            raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

        model.add_module(name, layer)

        if name in content_layers:
            # add content loss:
            target = model(content_img).detach()
            content_loss = ContentLoss(target)
            model.add_module("content_loss_{}".format(i), content_loss)
            content_losses.append(content_loss)

        if name in style_layers:
            # add style loss:
            target_feature = model(style_img).detach()
            style_loss = StyleLoss(target_feature)
            model.add_module("style_loss_{}".format(i), style_loss)
            style_losses.append(style_loss)

    # now we trim off the layers after the last content and style losses
    for i in range(len(model) - 1, -1, -1):
        if isinstance(model[i], ContentLoss) or isinstance(model[i], StyleLoss):
            break

    model = model[:(i + 1)]
    
    return model, style_losses, content_losses


"""Finally, we must define a function that performs the neural transfer. For
each iteration of the networks, it is fed an updated input and computes
new losses. We will run the ``backward`` methods of each loss module to
dynamicaly compute their gradients. The optimizer requires a “closure”
function, which reevaluates the module and returns the loss.

We still have one final constraint to address. The network may try to
optimize the input with values that exceed the 0 to 1 tensor range for
the image. We can address this by correcting the input values to be
between 0 to 1 each time the network is run.



"""


def run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True, num_steps=300,
                     style_weight=1000000, content_weight=1):
    """Run the image reconstruction, texture synthesis, or style transfer."""
    print('Building the style transfer model..')
    # get your model, style, and content losses
    model, style_losses, content_losses = get_model_and_losses(cnn, style_img, content_img)

    # get the optimizer
    input_img = input_img.requires_grad_()
    optimizer = get_image_optimizer(input_img)
    
    model = model.to(device)
    model.eval()
    model.requires_grad_(False)
    
    print('Optimizing..')
    
    curr_step = 0
    
    while curr_step <= num_steps:
        def closure():
            nonlocal curr_step
            # correct the values of updated input image
            input_img.data.clamp_(0, 1)
            
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            
            for sl in style_losses:
                style_score += sl.loss
            for cl in content_losses:
                content_score += cl.loss
            
            if use_style:
                style_score *= style_weight
            if use_content:
                content_score *= content_weight
            
            loss = style_score + content_score
            loss.backward()
            curr_step += 1
            if curr_step % 50 == 0:
                print("Step {}: Style Loss: {:4f} Content Loss: {:4f}".format(curr_step, style_score.item(), content_score.item()))
            
            return loss
        
        optimizer.step(closure)
    # run model training, with one weird caveat
    
    # we recommend you use LBFGS, an algorithm which preconditions the gradient
    # with an approximate Hessian taken from only gradient evaluations of the function
    # this means that the optimizer might call your function multiple times per step, so as
    # to numerically approximate the derivative of the gradients (the Hessian)
    # so you need to define a function
    # def closure():
    # here
    # which does the following:
    # clear the gradients
    # compute the loss and it's gradient
    # return the loss

    # one more hint: the images must be in the range [0, 1]
    # but the optimizer doesn't know that
    # so you will need to clamp the img values to be in that range after every step

    # make sure to clamp once you are done
    with torch.no_grad():
        input_img.data.clamp_(0, 1)

    return input_img


def main(style_img_path, content_img_path):
    print("Running Neural Style Transfer")
    # we've loaded the images for you
    style_img = load_image(style_img_path)
    content_img = load_image(content_img_path)
    
    style_img_name = style_img_path.split('/')[-1].split('.')[0]
    content_img_name = content_img_path.split('/')[-1].split('.')[0]

    print(f'Style Image Size: {style_img.size()}')
    print(f'Content Image Size: {content_img.size()}')
    # interative MPL
    plt.ion()
    
    # if images are not the same size, resize the style image to the content image size
    if style_img.size() != content_img.size():
        # resize the style image to the size of the content image
        style_img = torch.nn.functional.interpolate(style_img, size=content_img.shape[-2:], mode='area')
        print("Resizing Style Image to match Content Image Size")
        print(f'Resized Style Image Size: {style_img.size()}\n')
        if style_img.size() != content_img.size():
            print("Channel size mismatch between style and content images after resizing. Filling the extra channels with first channel value")
            style_img = style_img.expand(-1, 3, -1, -1)
            print(f'Resized Style Image Size: {style_img.size()}\n')
        
    assert style_img.size() == content_img.size(), \
        "we need to import style and content images of the same size"

    # plot the original input image:
    # plt.figure()
    # imshow(style_img, title='Style Image')

    # plt.figure()
    # imshow(content_img, title='Content Image')

    # we load a pretrained VGG19 model from the PyTorch models library
    # but only the feature extraction part (conv layers)
    # and configure it for evaluation
    cnn = models.vgg19(pretrained=True).features.to(device).eval()

    # image reconstruction
    print("Performing Image Reconstruction from white noise initialization")
    # random noise of the size of content_img on the correct device
    input_img = torch.randn(content_img.data.size(), device=device)
    # reconstruct the image from the noise
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=False)

    # save the image
    save_path = f'output/{content_img_name}_{style_img_name}_reconstructed.jpg'
    plt.imsave(save_path, output.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0))
    print(f"Reconstructed Image saved at {save_path}\n")
    # plt.figure()
    # imshow(output, title='Reconstructed Image')

    # texture synthesis
    print("Performing Texture Synthesis from white noise initialization")
    # random noise of the size of content_img on the correct device
    input_img = torch.randn(content_img.data.size(), device=device)
    # synthesize a texture like style_image
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=False, use_style=True)

    # save the image
    save_path = f'output/{content_img_name}_{style_img_name}_synthesized.jpg'
    plt.imsave(save_path, output.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0))
    print(f"Synthesized Image saved at {save_path}\n")
    # plt.figure()
    # imshow(output, title='Synthesized Texture')

    # style transfer
    # random noise of the size of content_img on the correct device
    input_img = torch.randn(content_img.data.size(), device=device)
    # transfer the style from the style_img to the content image
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True)

    save_path = f'output/{content_img_name}_{style_img_name}_styled.jpg'
    plt.imsave(save_path, output.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0))
    print(f"Styled Image saved at {save_path}\n")
    # plt.figure()
    # imshow(output, title='Output Image from noise')

    print("Performing Style Transfer from content image initialization")
    input_img = content_img.clone()
    # transfer the style from the style_img to the content image
    output = run_optimization(cnn, content_img, style_img, input_img, use_content=True, use_style=True)

    # save the image
    save_path = f'output/{content_img_name}_{style_img_name}_styled_content.jpg'
    plt.imsave(save_path, output.squeeze(0).cpu().detach().numpy().transpose(1, 2, 0))
    print(f"Styled Image saved at {save_path}\n")
    # plt.figure()
    # imshow(output, title='Output Image from noise')

    plt.ioff()
    plt.show()


if __name__ == '__main__':
    # Example: python run.py images/style/escher_sphere.jpeg images/content/dancing.jpg
    # args = sys.argv[1:3]
    # main(*args)
    
    content_folder = 'images/content/'
    style_folder = 'images/style/'
    
    for content_img in os.listdir(content_folder):
        for style_img in os.listdir(style_folder):
            print(f"----------------------------------\nContent Image: {content_img}\nStyle Image: {style_img}\n")
            main(style_folder + style_img, content_folder + content_img)
            print(f"----------------------------------\n")
