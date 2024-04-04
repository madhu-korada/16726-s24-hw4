# Assignment #4 - Neural Style Transfer

In this assignment, I will implement neural style transfer which resembles specific content in a certain artistic style. For example, generate cat images in Ukiyo-e style. The algorithm takes in a content image, a style image, and another input image. The input image is optimized to match the previous two target images in content and style distance space.

## Part 1: Content Reconstruction 

#### Effect of optimizing content loss at different layers: 

|                        Original Image                        |                       layer: `conv_2`                        |                       layer: `conv_4`                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|               ![](images/content/dancing.jpg)                | ![](output/dancing_escher_sphere/dancing_escher_sphere_reconstructed_conv_2.jpg) | ![](output/dancing_escher_sphere/dancing_escher_sphere_reconstructed_conv_4.jpg) |
|                     **layer: `conv_6`**                      |                     **layer: `conv_7`**                      |                     **layer: `conv_8`**                      |
| ![](output/dancing_escher_sphere/dancing_escher_sphere_reconstructed_conv_6.jpg) | ![](output/dancing_escher_sphere/dancing_escher_sphere_reconstructed_conv_7.jpg) | ![](output/dancing_escher_sphere/dancing_escher_sphere_reconstructed_conv_8.jpg) |
|                     **layer: `conv_10`**                     |                     **layer: `conv_12`**                     |                     **layer: `conv_14`**                     |
| ![](output/dancing_escher_sphere/dancing_escher_sphere_reconstructed_conv_10.jpg) | ![](output/dancing_escher_sphere/dancing_escher_sphere_reconstructed_conv_12.jpg) | ![](output/dancing_escher_sphere/dancing_escher_sphere_reconstructed_conv_14.jpg) |

Here the conv naming convention is defined in a way that the smaller the number the closer it is to the end of the network. i.e, `conv_2` is last second conv layer of the VGG. 

```
------------------- Model Layers -------------------
Name: conv_1,  Layer: Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_1,  Layer: ReLU()
Name: conv_2,  Layer: Conv2d(64, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_2,  Layer: ReLU()
Name: pool_2,  Layer: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
Name: conv_3,  Layer: Conv2d(64, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_3,  Layer: ReLU()
Name: conv_4,  Layer: Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_4,  Layer: ReLU()
Name: pool_4,  Layer: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
Name: conv_5,  Layer: Conv2d(128, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_5,  Layer: ReLU()
Name: conv_6,  Layer: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_6,  Layer: ReLU()
Name: conv_7,  Layer: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_7,  Layer: ReLU()
Name: conv_8,  Layer: Conv2d(256, 256, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_8,  Layer: ReLU()
Name: pool_8,  Layer: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
Name: conv_9,  Layer: Conv2d(256, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_9,  Layer: ReLU()
Name: conv_10, Layer: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_10, Layer: ReLU()
Name: conv_11, Layer: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_11, Layer: ReLU()
Name: conv_12, Layer: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_12, Layer: ReLU()
Name: pool_12, Layer: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
Name: conv_13, Layer: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_13, Layer: ReLU()
Name: conv_14, Layer: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_14, Layer: ReLU()
Name: conv_15, Layer: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_15, Layer: ReLU()
Name: conv_16, Layer: Conv2d(512, 512, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))
Name: relu_16, Layer: ReLU()
Name: pool_16, Layer: MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
------------------------------------------------------
```

As we can see, content loss after `conv_2` and `conv_4` work the best. 



#### Take two random noises as two input images, optimize them only with content loss: 

|                            Noise                             |                     Reconstructed Image                      |                        Content Image                         |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| ![](output/tubingen_picasso/tubingen_picasso_noise_conv_4.jpg) | ![](output/tubingen_picasso/tubingen_picasso_reconstructed_conv_4.jpg) | <img src="images/content/tubingen.jpeg" style="zoom:80%;" /> |
|   ![](output/wally_picasso/wally_picasso_noise_conv_4.jpg)   | ![](output/wally_picasso/wally_picasso_reconstructed_conv_4.jpg) |  <img src="images/content/wally.jpg" style="zoom: 25%;" />   |



## Part 2: Texture Synthesis 

#### Effect of optimizing texture loss at different layers. 

Looking at the below results I used `conv_1`, `conv_2`, `conv_3`, `conv_4`, `conv_5`.

|                        Original Image                        |                   layer: `conv 1,2,3,4,5`                    |                   layer: `conv 1,2,4,7,11`                   |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|  <img src="images/style/picasso.jpg" style="zoom: 67%;" />   | ![](output/fallingwater_picasso/fallingwater_picasso_synthesized_conv_1_2_3_4_5.jpg) | ![](output/fallingwater_picasso/fallingwater_picasso_synthesized_conv_1_2_4_7_11.jpg) |
|                 **layer: `conv 1,3,5,7,9`**                  |                 **layer: `conv 2,4,6,8,10`**                 |                **layer: `conv 3,6,9,12,15`**                 |
| ![](output/fallingwater_picasso/fallingwater_picasso_synthesized_conv_1_3_5_7_9.jpg) | ![](output/fallingwater_picasso/fallingwater_picasso_synthesized_conv_2_4_6_8_10.jpg) | ![](output/fallingwater_picasso/fallingwater_picasso_synthesized_conv_3_6_9_12_15.jpg) |

#### Random noises as two input images, optimize them only with style loss. 


|                            Noise                             |                      Synthesized Image                       |                       Style Image                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :------------------------------------------------------: |
| ![](output/fallingwater_picasso/fallingwater_picasso_noise_conv_1_2_3_4_5.jpg) | ![](output/fallingwater_picasso/fallingwater_picasso_synthesized_conv_1_2_3_4_5.jpg) | <img src="images/style/picasso.jpg" style="zoom:80%;" /> |
| ![](output/fallingwater_picasso/fallingwater_picasso_noise1_conv_1_2_3_4_5.jpg) | ![](output/fallingwater_picasso/fallingwater_picasso_noise1_synthesized_conv_1_2_3_4_5.jpg) | <img src="images/style/picasso.jpg" style="zoom:80%;" /> |



## Part 3: Style Transfer

#### Hyper-parameters Tuning

|      Hyper Prameters      |                    Random Initialization                     |                 Content based Initialization                 |
| :-----------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
|   **`style weight: 1`**   | ![](output/wally_starry_night/wally_starry_night_styled_sw_1_cw_1.jpg) | ![](output/wally_starry_night/wally_starry_night_styled_content_sw_1_cw_1.jpg) |
| **`style weight : 1000`** | ![](output/wally_starry_night/wally_starry_night_styled_sw_1000_cw_1.jpg) | ![](output/wally_starry_night/wally_starry_night_styled_content_sw_1000_cw_1.jpg) |
|  **`style weight: 1e6`**  | ![](output/wally_starry_night/wally_starry_night_styled_sw_1000000_cw_1.jpg) | ![](output/wally_starry_night/wally_starry_night_styled_content_sw_1000000_cw_1.jpg) |
|  **`style weight: 1e9`**  | ![](output/wally_starry_night/wally_starry_night_styled_sw_1000000000_cw_1.jpg) | ![](output/wally_starry_night/wally_starry_night_styled_content_sw_1000000000_cw_1.jpg) |
| **`style weight: 1e12`**  | ![](output/wally_starry_night/wally_starry_night_styled_sw_1000000000000_cw_1.jpg) | ![](output/wally_starry_night/wally_starry_night_styled_content_sw_1000000000000_cw_1.jpg) |

As we can see content based initialization with style weight 100000 gives us the best result. 



#### More Style transfer results: 

|                       Style Images >>                        | <img src="images/style/escher_sphere.jpeg" style="zoom:25%;" /> |           ![](images/style/frida_kahlo.jpeg)            | <img src="images/style/the_scream.jpeg" style="zoom:33%;" /> |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :-----------------------------------------------------: | ------------------------------------------------------------ |
|                      **Content Images**                      |                                                              |                                                         |                                                              |
| <img src="images/content/dancing.jpg" style="zoom: 67%;" />  |     ![](output/dancing_escher_sphere_styled_content.jpg)     |   ![](output/dancing_frida_kahlo_styled_content.jpg)    | ![](output/dancing_the_scream_styled_content.jpg)            |
| <img src="images/content/phipps.jpeg" style="zoom: 50%;" />  |     ![](output/phipps_escher_sphere_styled_content.jpg)      |    ![](output/phipps_frida_kahlo_styled_content.jpg)    | ![](output/phipps_the_scream_styled_content.jpg)             |
| <img src="images/content/fallingwater.png" style="zoom: 50%;" /> |  ![](output/fallingwater_escher_sphere_styled_content.jpg)   | ![](output/fallingwater_frida_kahlo_styled_content.jpg) | ![](output/fallingwater_the_scream_styled_content.jpg)       |
| <img src="images/content/tubingen.jpeg" style="zoom: 50%;" /> |    ![](output/tubingen_escher_sphere_styled_content.jpg)     |   ![](output/tubingen_frida_kahlo_styled_content.jpg)   | ![](output/tubingen_the_scream_styled_content.jpg)           |



#### Random Noise VS Content Image

|                        Content Image                         | Style Image                   |            Random Initialization            |            Content based Initialization             |
| :----------------------------------------------------------: | ----------------------------- | :-----------------------------------------: | :-------------------------------------------------: |
| <img src="images/content/fallingwater.png" style="zoom: 50%;" /> | ![](images/style/picasso.jpg) | ![](output/fallingwater_picasso_styled.jpg) | ![](output/fallingwater_picasso_styled_content.jpg) |
|                                                              |                               |       **`Inference time: 16.5 secs`**       |          **`Inference time: 11.23 secs`**           |



#### Style transfer on some of your favourite images. 


|                        Content Image                         |                         Style Image                          |                        Stylized Image                        |
| :----------------------------------------------------------: | :----------------------------------------------------------: | :----------------------------------------------------------: |
| <img src="images/content/cinderella_an.jpg" style="zoom:90%;" /> | <img src="images/style/the_scream.jpeg" style="zoom:50%;" /> | ![](output/cinderella_an_the_scream/cinderella_an_the_scream_styled_content_sw_1000000_cw_1.jpg) |
| <img src="images/content/pixels.jpg" alt="pixels" style="zoom:20%;" /> |       ![starry_night](images/style/starry_night.jpeg)        | ![](output/pixels_starry_night/pixels_starry_night_styled_content_sw_1000000_cw_1.jpg) |





## Bells & Whistles (Extra Points)

### Stylized grump cats 

|        Content Image        |            Style Image             |                        Stylized Image                        |
| :-------------------------: | :--------------------------------: | :----------------------------------------------------------: |
| ![](images/content/cat.png) | ![](images/style/frida_kahlo.jpeg) | ![](output/cat_frida_kahlo/cat_frida_kahlo_styled_content_sw_1000000_cw_1.jpg) |


