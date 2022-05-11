# Image-Super-Resolution-Using-Conv-Nets

Single Image Super-Resolution is a classical Computer Vision inverse mapping problem that reconstructs/recovers a High Resolution (HR) image from a Low Resolution (LR) image. The solution to Super Resolution task is not unique. Modern Deep Learning Systems using Conv Nets have attained a large popularity in solving this task through generating super resolution patches which can be considered as generating a mapping from Low Resolution domain to Higher Resolution domain. Utilizing various deep CNN systems,in a supervised learning approach, we aim to reconstruct Higher Resolution (HR) images from Lower Resolution (LR) images. We make use of different criterion functions to have the best perceptual quality in the generate super resolution images. We have PSNR and SSIM metrics for evaluation of our model performance on unseen data.  The primary purpose of this project is to produce quality High Resolution images from their Low Resolution counterpart.
```
|_
  |_ SRCNN/
     |_ run_srcnn.ipynb
     |_ Arun_srcnn.ipynb
     |_ model.py
     |_ dataset.py
     |_ Adataset.py
     |_losses.py
  |_ SRGAN/
     |_ 541-srgan.ipynb
     |_ 541-srresnet.ipynb
     |_ model.py
     |_ dataset.py
     |_ losses.py
  |_ downloaddiv2k.py
  |_ augument.ipynb
  |_ DataPreProcessing.ipynb
  |_ 541_psnr_ssim.ipynb.
  |_ results
  |_ README.md
  ```

# Setup

We trainied different architectures for generating Super Resolution Images on Google Colab with Tesla K80 with 12 GB RAM

## Installation Instructions
- Download Dataset using downloaddiv2k.py  | Thanks to https://github.com/Lornatang/SRGAN-PyTorch/blob/master/scripts/prepare_dataset.py for the script
- Preporcess the data using DataPreprocessing.ipynb and augument.ipynb
- To start training and testing our models we provide a single run_* ipyninb file in each directory which will be used for training, validation and testing purposes

# Code Structure
- SRCNN Directory Contains Models for SRCNN and DRRN.
- SRGAN Directory COntains Models for SRResNet and SRGAN
- Results directorys contains few sample image outputs with PSNR and SSIM values for comparison of our models

# Results

<p float="center">
  <img src="https://github.com/sudujr/Image-Super-Resolution-Using-Conv-Nets/blob/main/results/result-srcnn.png" width="100%" />

</p>

<p float="center">
  <img src="https://github.com/sudujr/Image-Super-Resolution-Using-Conv-Nets/blob/main/results/result-drrn.png" width="100%" />

</p>

<p float="center">
  <img src="https://github.com/sudujr/Image-Super-Resolution-Using-Conv-Nets/blob/main/results/result-srresnet.png" width="100%" />

</p>

<p float="center">
  <img src="https://github.com/sudujr/Image-Super-Resolution-Using-Conv-Nets/blob/main/results/result-srgan.png" width="100%" />

</p>

# References

		\bibitem{SRCNN} Dong, Chao, et al., \href{https://arxiv.org/pdf/1501.00092.pdf}{Image super-resolution using deep convolutional networks.}, in IEEE transactions on pattern analysis and machine intelligence 38.2 (2015): 295-307.
	
	
	\bibitem{DRRN} Tai, Ying, Jian Yang, and Xiaoming Liu., \href{https://openaccess.thecvf.com/content_cvpr_2017/papers/Tai_Image_Super-Resolution_via_CVPR_2017_paper.pdf}{Image super-resolution via the deep recursive residual network.}, in Proceedings of the IEEE conference on computer vision and pattern recognition. 2017.

	\bibitem{SRGAN} Ledig, Christian, et al., \href{https://arxiv.org/pdf/1609.04802.pdf}{Photo-realistic single image super-resolution using a generative adversarial network.}, in Proceedings of the IEEE conference on computer vision and pattern recognition. 2017
	
	\bibitem{Survey} Wang, Zhihao, Jian Chen, and Steven CH Hoi., \href{https://arxiv.org/pdf/1902.06068.pdf}{Deep learning for image super-resolution: A survey.}, in IEEE transactions on pattern analysis and machine intelligence 43.10 (2020): 3365-3387.
	
	\bibitem{ResNet} He, Kaiming, et al., \href{https://arxiv.org/pdf/1512.03385.pdf}{Deep residual learning for image recognition.}, in Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
	
	\bibitem{RecursiveNets} Kim, Jiwon, Jung Kwon Lee, and Kyoung Mu Lee., \href{https://arxiv.org/pdf/1511.04491.pdf}{Deeply-recursive convolutional network for image super-resolution.}, in Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
	
	\bibitem{GAN} Goodfellow, Ian, et al., \href{https://arxiv.org/pdf/1406.2661.pdf}{Generative adversarial nets.}, in Proceedings of Advances in neural information processing systems 27 (2014).	

	\bibitem{CGAN} Mirza, Mehdi, and Simon Osindero., \href{https://arxiv.org/pdf/1411.1784.pdf}{Conditional generative adversarial nets.}, in arXiv preprint arXiv:1411.1784 (2014).	
	
	\bibitem{DCGAN} Radford, Alec, Luke Metz, and Soumith Chintala., \href{https://arxiv.org/pdf/1511.06434.pdf}{Unsupervised representation learning with deep convolutional generative adversarial networks.}, in arXiv preprint arXiv:1511.06434 (2015).	
	

	\bibitem{NITRE 2017} Agustsson, Eirikur, and Radu Timofte., \href{https://ieeexplore.ieee.org/document/8014884}{Ntire 2017 challenge on single image super-resolution: Dataset and study.}, in Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2017.
	
	\bibitem{NITRE 2018} Timofte, Radu, et al., \href{https://openaccess.thecvf.com/content_cvpr_2018_workshops/papers/w13/Timofte_NTIRE_2018_Challenge_CVPR_2018_paper.pdf}{Ntire 2018 challenge on single image super-resolution: Methods and results.}, in Proceedings of the IEEE conference on computer vision and pattern recognition workshops. 2018.
	
	\bibitem{PIRM} Ignatov, Andrey, et al., \href{https://openaccess.thecvf.com/content_ECCVW_2018/papers/11133/Ignatov_PIRM_Challenge_on_Perceptual_Image_Enhancement_on_Smartphones_Report_ECCVW_2018_paper.pdf}{Pirm challenge on perceptual image enhancement on smartphones: Report.}, in Proceedings of the European Conference on Computer Vision (ECCV) Workshops. 2018.
	
	\bibitem{CutBlur} Han, Junlin, et al., \href{https://arxiv.org/pdf/2201.12078.pdf}{You Only Cut Once: Boosting Data Augmentation with a Single Cut.}, in arXiv preprint arXiv:2201.12078 (2022).

	\bibitem{RGB} Yoo, Jaejun, Namhyuk Ahn, and Kyung-Ah Sohn., \href{https://arxiv.org/pdf/2201.12078.pdf}{Rethinking data augmentation for image super-resolution: A comprehensive analysis and a new strategy.}, in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2020.
	
	
	
	\bibitem{CODE BASE 1} \textit{SRCNN}, available at \url{https://github.com/kunal-visoulia/Image-Restoration-using-SRCNN}
	
	\bibitem{CODE BASE 2} \textit{DRRN}, available at \url{https://github.com/yjn870/DRRN-pytorch}
    \bibitem{CODE BASE 2} \textit{SRGAN}, available at \url{https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Super-Resolution}
	
	\bibitem{CODE BASE 2} \textit{SRGAN}, available at \url{https://github.com/leftthomas/SRGAN}
