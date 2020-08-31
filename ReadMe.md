# [WIP] retinaface4j
## Because not all production environements can be in python
RetinaFace (RetinaFace: Single-stage Dense Face Localisation in the Wild, published in 2019), in java.  
This repo is an experiment attempting to answer the question : Is pytorch java + ND4J a viable option for deep learning on the JVM ?  

Original paper -> [arXiv](https://arxiv.org/pdf/1905.00641.pdf)  
Original Mxnet implementation -> [Insightface](https://github.com/deepinsight/insightface/tree/master/RetinaFace)  
Insprired by the Pytorch implementation -> [Pytorch_Retinaface](https://github.com/biubug6/Pytorch_Retinaface)

### Table of contents
1. [ Installation ](#Installation)
2. [ Usage ](#Usage)
3. [ Acknowledgements ](#Acknowledgements)

example output : 
![testing on a random internet selfie](output.png)

*****
<a name="Installation"></a>
## INSTALLATION
TODO 

<a name="Usage"></a>
## USAGE
Download pretrained weights on [Dropbox](https://www.dropbox.com/sh/ar7q20icorvpaxi/AADFuPlnQEe78nsnGA7wzwuoa?dl=0)  
Run  :
```angular2
./gradlew run --args="./weights/Retinaface_resnet_traced.pt ./sample-images/WC_FR.jpeg output.png"
```
Java usage :
```java
import java.util.*;
import javax.imageio.ImageIO;
import java.io.File;
import java.io.IOException;
import retinaface4j.Detector;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.datavec.image.loader.NativeImageLoader;

String modelPath = "./models/Retinaface_resnet_traced.pt";
String imgPath = "./sample-images/WC_FR.jpeg";
Double detThresh = 0.9;
Double nmsThresh = 0.4;
BufferedImage img = null;
INDArray nd4jimg = null;
try {
    img = ImageIO.read(new File(imgPath));
    NativeImageLoader loader = new NativeImageLoader(img.getHeight(), img.getWidth(), 3);
    nd4jimg = loader.asMatrix(img);
} catch (IOException e) {
    System.out.println("[ERROR] could not read image");
    System.out.println(e.getMessage());
}

Detector detector = new Detector(modelPath, detThresh, nmsThresh);
INDArray dets = detector.predict(nd4jimg);
```


## ACKNOWLEDGEMENTS
This work is laergely based on :  
The original implementation by the [insightface](https://github.com/deepinsight/insightface) team.  
The [pyton pytorch implementation](https://github.com/biubug6/Pytorch_Retinaface).
If you use this repo, please reference the original work :  

```  
@inproceedings{Deng2020CVPR,
title = {RetinaFace: Single-Shot Multi-Level Face Localisation in the Wild},
author = {Deng, Jiankang and Guo, Jia and Ververas, Evangelos and Kotsia, Irene and Zafeiriou, Stefanos},
booktitle = {CVPR},
year = {2020}
}
```
