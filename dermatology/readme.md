## Skin Cancer MNIST: HAM10000
This dataset contains 10,000 images of different dermastopic images with labels attached to them. The classes contained within the dataset were composed of:
* Actinic keratoses and intraepithelial carcinoma / Bowen's disease (akiec)
* Basal cell carcinoma (bcc) 
* Benign keratosis-like lesions (solar lentigines / seborrheic keratoses and lichen-planus like keratoses, bkl)
* Dermatofibroma (df)
* Melanoma (mel)
* Melanocytic nevi (nv) 
* vascular lesions (angiomas, angiokeratomas, pyogenic granulomas and hemorrhage, vasc). 


| Class    | Image Count| Proportion |
|----------|----------|----------|
| akiec  | 6705     | 0.669    |
| bcc   | 1113     |0.111     |
| bkl   | 1099    | 0.110   |
| df    | 514    | 0.051     |
| mel    | 327     | 0.033   |
| nv   | 142    |  0.014     |
| vasc    | 115     |0.011     |

Upon viewing the dataset, I made a goal to make a multiclass classification program which utlized a CNN created through PyTorch. Throughout my experiments, I made changes (not all cataloged) to my program to better classify between groups.

## Experiments:
