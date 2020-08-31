package retinaface4j;

import java.util.*;
import java.lang.*;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;
import org.datavec.image.loader.NativeImageLoader;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.net.URL;
import java.io.File;
import java.nio.file.Files;
import org.apache.commons.io.IOUtils;
import java.io.IOException;
import java.util.Arrays;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.awt.image.WritableRaster;
import java.util.Map;
import java.util.HashMap;

public class Detector{

    public static float[] pixelMeans = {104,117,123};

    private double nmsThresh;
    private double detThresh;
    private Module model;

    public Detector(double detThresh, double nmsThresh){
        String resource = Detector.class.getResource("/models/RetinaFace_resnet50_traced.pt").getPath();
        Module model = Module.load(resource);
        this.setModel(model);
        this.setDetThresh(detThresh);
        this.setNmsThresh(nmsThresh);
    }

    public void setModel(Module m) {
        this.model = m;
    }
    public void setDetThresh(double t){
        this.detThresh = t;
    }
    public void setNmsThresh(double t){
        this.nmsThresh = t;
    }

    public INDArray predict(INDArray img){

        INDArray inputImg = Utils.substractMean(img, this.pixelMeans);


        // Run inference
        Tensor data =
            Tensor.fromBlob(
                inputImg.data().asNioFloat(),
                new long[] {1,3,inputImg.shape()[2],inputImg.shape()[3]} // shape
                );
        IValue result = this.model.forward(IValue.from(data));
        IValue[] output = result.toTuple();
        Tensor loc = output[0].toTensor();
        Tensor conf = output[1].toTensor();
        Tensor landmks = output[2].toTensor();


        // Post processing
        PriorBox pbox = new PriorBox((int) inputImg.shape()[3],(int) inputImg.shape()[2]);
        Tensor priors = pbox.forward();
        // boxes post proc
        INDArray boxes = Utils.decode(loc.getDataAsFloatArray(), priors.getDataAsFloatArray(), loc.shape());
        long[] shape = {inputImg.shape()[3], inputImg.shape()[2], inputImg.shape()[3], inputImg.shape()[2]};
        INDArray nd4jShape = Nd4j.createFromArray(shape);
        boxes = boxes.mul(nd4jShape);
        // confidences post proc
        long[] confShape = conf.shape();
        long[] confShapeSqueezed = {confShape[1], confShape[2]};
        INDArray nd4jConf = Nd4j.createFromArray(conf.getDataAsFloatArray()).reshape(confShapeSqueezed);
        INDArray scores = nd4jConf.getColumn(1);
        // landmks post proc
        INDArray nd4jLandmks = Utils.decode_landm(landmks.getDataAsFloatArray(), priors.getDataAsFloatArray(), landmks.shape());
        long[] scale1 = {inputImg.shape()[3], inputImg.shape()[2], inputImg.shape()[3], inputImg.shape()[2], inputImg.shape()[3], inputImg.shape()[2], inputImg.shape()[3], inputImg.shape()[2], inputImg.shape()[3], inputImg.shape()[2]};
        INDArray nd4jScale1 = Nd4j.createFromArray(scale1);
        nd4jLandmks = nd4jLandmks.mul(nd4jScale1);


        // ignore low scores
        List<Integer> indices = new ArrayList<Integer>();
        for (int i =0; i<scores.shape()[0]; i++){
            if (scores.getDouble(i)>this.detThresh){
                indices.add(i);
            }
        }
        int[] indicesArr = new int[indices.size()];
        for (int i=0; i < indicesArr.length; i++)
        {
            indicesArr[i] = indices.get(i).intValue();
        }
        INDArray nd4jInds = Nd4j.createFromArray(indicesArr);
        scores = scores.get(nd4jInds);
        boxes = boxes.get(nd4jInds);
        nd4jLandmks = nd4jLandmks.get(nd4jInds);


        // NMS
        double thresh = 0.4;
        int[] thresholdedScroresUnsqueezedShape = {(int) scores.shape()[0], 1};
        INDArray dets = Nd4j.hstack(boxes, scores.reshape(thresholdedScroresUnsqueezedShape));
        INDArray keep = Utils.cpu_nms(dets, this.nmsThresh);
        INDArray dets2 = dets.getRow(keep.getInt(0));
        for (int i=1; i<keep.shape()[0]; i++) {
            long[] shapes = {1, dets.getRow(keep.getInt(i)).shape()[0]};
            dets2 = Nd4j.vstack(dets2,dets.getRow(keep.getInt(i)).reshape(shapes));
        }
        nd4jLandmks = nd4jLandmks.get(keep);
        dets = Nd4j.hstack(dets2, nd4jLandmks);


        return dets;
    }

}