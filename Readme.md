<b>** Open Project Report for Details</b>

Sample command to generate super resolution from given LR image:

	python generate_super_resolution.py --input_image army.jpg --model model_epoch_10.pth --output_filename super_army.jpg --cuda

Sample command to generate model based on paramenters as follows:

	python generate_model.py --upscale_factor 3 --trainBatchSize 4 --testBatchSize 100 --nEpochs 15 --lr 0.001 –-cuda --threads 4 --seed 123


Instructions to run the programs:

	- Keep all model.py data.py dataset.py generate_model.py generate_super_resolution.py files in same directory
	- Open terminal and run command: python3 generate_model.py -h
	- It gives the argument information on which the model can be trained. For e.g. –
	   python3 generate_model.py --upscale_factor 3 --trainBatchSize 4 --testBatchSize 100 --nEpochs 10 --lr 0.001 –-cuda --threads 4 --seed 123
	- The program will download the dataset from the internet and starts training on it
	- The program saves the model after each epoch in same directory with different name
	- The saved model then can be used to generate the super resolution image from any given lower resolution image.
	- Keep input low resolution image in same directory as above programs.
	- Run command: python3 generate_super_resolution.py -h
	- It gives the argument information. For e.g. – python generate_super_resolution.py --input_image army.jpg --model model_epoch_10.pth --output_filename super_army.jpg --cuda
	- The program will generate the super resolution image and saves in same directory with output name as provided in the argument. The Output image will be upscaled by trained model upscale_factor.

Note: For upscaling factor 1, GPU vram runs out hence it is trained on CPU for 10 epochs only.


Observations and Inferences:

When the model is run for 1 to 10 scaling factors, each 40 epochs (except scaling factor 1 in which epochs are 10 only because of it runs on CPU). The PSNR do not get increases after the 15-20 epochs, because the data generated from low resolution is enough for upscaling up to 3 to 5 times only, and after that, the model generates data but it does not significantly increases the details of output image, instead it just increases the resolution of image. As the image get more cleaner, the redundant pixels values are vanishing and hence the size of image is decreasing accordingly. Hence, after certain limit of SR, image dimension increases but it will not get good quality output in visual better than upscaling factor 4-5.


References:

1] Real‐Time Single Image and Video Super‐Resolution Using an Efficient Sub‐Pixel Convolutional Neural Network – Shi et.al ‐ https://arxiv.org/abs/1609.05158

2] Wikipedia ‐ https://www.wikipedia.org/

3] Pytorch documentation ‐ https://pytorch.org/docs/stable/nn.html

4] Github for debugging and code snippets - https://github.com/

5] Debugging and code snippets - https://stackoverflow.com/
