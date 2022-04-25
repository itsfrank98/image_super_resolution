# image_super_resolution
Attempt to real world image super resolution using LapSRN. Perceptual loss was tried (VGGLoss.py) but in the end not used because of bad performance
and inefficiency.
Subpixel convolution is used as upsampling technique.
Python 3.9

To run the project:
* install the dependencies by running ```pip install -r requirements.txt```
* preprocess the images (more info in the file ```utils.py```). Preprocessing consists in:
  * scaling the images to the 0-1 range
  * converting the images from RGB to YCbCr and considering only the Y channel (common practice)
  * from each LR image, take 100 patches of size 128x128
  * for each patch, take the correspondant HR patch from the HR image. IMPORTANT: to do so, I had to make a small modification to the implementation of sklearn.feature_extraction. You need to go to the path ```venv/lib/python3.9/site-packages/sklearn/feature_extraction/image.py``` and replace the return statement at row 404 with the following: ```return patches, i_s, j_s```
* train the model by running the main in the file ```train.py``` (change the parameters accordingly to your needs, I trained two models: one for x2 SR, one for x4 SR)
* test your model by computing the PSNR values (more info in the file ```testing.py```)
* you can also run a demo to upscale an arbitrary image (```demo.py```)

## Dataset structure
The image matrices are put together in a unique ```.npy``` file. train_hr and train_lr contain the LR and HR images used in the training phase. test_hr and test_lr are the LR-HR images used to perform the evaluation metrics in the testing phase, while valid_hr and valid_lr contain images for the validation
```
dataset
- x2
  -- test_hr.npy
  -- train_hr.npy
  -- test_lr.npy
  -- train_lr.npy
  -- valid_hr.npy
  -- valid_lr.npy
- x4
  -- test_hr.npy
  -- train_hr.npy
  -- test_lr.npy
  -- train_lr.npy
  -- valid_hr.npy
  -- valid_lr.npy
```


# Credits:
KONG, Lei, et al. An Improved Image Super-Resolution Reconstruction Method Based On LapSRN. In: 2021 14th International Congress on Image and Signal Processing, BioMedical Engineering and Informatics (CISP-BMEI). IEEE, 2021. p. 1-5.
