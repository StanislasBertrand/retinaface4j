package retinaface4j;

import java.util.*;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.io.File;
import java.io.IOException;
import java.util.Arrays;
import java.awt.image.BufferedImage;
import javax.imageio.ImageIO;
import java.awt.image.WritableRaster;
import java.util.Map;
import java.util.HashMap;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.indexing.BooleanIndexing;
import org.nd4j.linalg.indexing.NDArrayIndex;

import retinaface4j.ArrayUtils;
import retinaface4j.PriorBox;
import retinaface4j.Utils;

public class App {
    public static void main(String[] args) {
        Module mod = Module.load("/home/bertrans/projects/personnal/Pytorch_Retinaface/weights/RetinaFace_resnet50_traced.pt");

        BufferedImage img = null;
        try {
//             img = ImageIO.read(new File("/home/bertrans/Documents/hey.jpg"));
            img = ImageIO.read(new File("/home/bertrans/projects/personnal/RetinaFace-tf2/sample-images/WC_FR.jpeg"));
        } catch (IOException e) {
            System.out.println("rip");
        }

        Map<Integer, Float> pixel_avg = new HashMap<>() {
            {
                put(0, 104f);
                put(1, 117f);
                put(2, 123f);
            }
        };

        WritableRaster wr = img.getRaster();
        float[] pixels = new float[3*wr.getWidth()*wr.getHeight()];


        for (int i = 0; i < wr.getWidth(); i++) {
            for (int j = 0; j < wr.getHeight(); j++) {
                for (int k = 0; k < 3; k++) {
                    if (k==0) {
                        int l = 2;
                        pixels[i + wr.getWidth()*j + (wr.getWidth() * wr.getHeight() * l)] = wr.getSample(i, j, k) - pixel_avg.get(l);
                    } else if (k==1) {
                        int l = 1;
                        pixels[i + wr.getWidth()*j + (wr.getWidth() * wr.getHeight() * l)] = wr.getSample(i, j, k) - pixel_avg.get(l);
                    } else if (k==2) {
                        int l = 0;
                        pixels[i + wr.getWidth()*j + (wr.getWidth() * wr.getHeight() * l)] = wr.getSample(i, j, k) - pixel_avg.get(l);
                    }

                }
            }
        }

    Tensor data =
        Tensor.fromBlob(
            pixels,
            new long[] {1,3,wr.getHeight(),wr.getWidth()} // shape
            );

    IValue result = mod.forward(IValue.from(data));
    IValue[] output = result.toTuple();
    Tensor loc = output[0].toTensor();
    Tensor conf = output[1].toTensor();
    Tensor landms = output[2].toTensor();

    PriorBox pbox = new PriorBox(wr.getWidth(), wr.getHeight());
    Tensor priors = pbox.forward();

    long[] locShape = loc.shape();
    long[] priorsShape = priors.shape();

    INDArray boxes = Utils.decode(loc.getDataAsFloatArray(), priors.getDataAsFloatArray(), locShape);
    long[] shape = {wr.getWidth(), wr.getHeight(), wr.getWidth(), wr.getHeight()};
    INDArray nd4jShape = Nd4j.createFromArray(shape);
    INDArray boxes2 = boxes.mul(nd4jShape);

    long[] confShape = conf.shape();
    long[] confShapeSqueezed = {confShape[1], confShape[2]};
    INDArray nd4jConf = Nd4j.createFromArray(conf.getDataAsFloatArray()).reshape(confShapeSqueezed);
    INDArray nd4jScores = nd4jConf.getColumn(1);

    long[] landnmsShape = landms.shape();
    INDArray landnms = Utils.decode_landm(landms.getDataAsFloatArray(), priors.getDataAsFloatArray(), landnmsShape);

    long[] scale1 = {wr.getWidth(), wr.getHeight(), wr.getWidth(), wr.getHeight(), wr.getWidth(), wr.getHeight(), wr.getWidth(), wr.getHeight(), wr.getWidth(), wr.getHeight()};
    INDArray nd4jScale1 = Nd4j.createFromArray(scale1);
    INDArray landnms2 = landnms.mul(nd4jScale1);

    // ignore low scores
    List<Integer> indices = new ArrayList<Integer>();
    for (int i =0; i<nd4jScores.shape()[0]; i++){
        if (nd4jScores.getDouble(i)>0.9){
            indices.add(i);
        }
    }
    int[] indicesArr = new int[indices.size()];
    for (int i=0; i < indicesArr.length; i++)
    {
        indicesArr[i] = indices.get(i).intValue();
    }
    INDArray nd4jInds = Nd4j.createFromArray(indicesArr);

    INDArray thresholdedScrores = nd4jScores.get(nd4jInds);
    INDArray thresholdedBoxes = boxes2.get(nd4jInds);
    INDArray thresholdedLandnms = landnms2.get(nd4jInds);

    // keep top-K before NMS TO CHECK VALUES AGAINS AND KEEP GOING WITH BOXES3 ETC.
//     INDArray order = Nd4j.reverse(Nd4j.createFromArray(ArrayUtils.argsort(thresholdedScrores.toFloatVector())));
//     int maxInd = 750;
//     if (order.shape()[0] < maxInd){
//         maxInd = (int) order.shape()[0];
//     }
//     INDArray orderTop = order.get(NDArrayIndex.interval(0,maxInd));
//     INDArray scores3 = thresholdedScrores.get(orderTop);
//     INDArray boxes3 = thresholdedBoxes.get(orderTop);
//     INDArray landnms3 = thresholdedLandnms.get(orderTop);


    // NMS
    double thresh = 0.4;
    int[] thresholdedScroresUnsqueezedShape = {(int) thresholdedScrores.shape()[0], 1};
    INDArray dets = Nd4j.hstack(thresholdedBoxes, thresholdedScrores.reshape(thresholdedScroresUnsqueezedShape));
    INDArray keep = Utils.cpu_nms(dets, thresh);

    INDArray dets2 = dets.getRow(keep.getInt(0));

    for (int i=1; i<keep.shape()[0]; i++) {
        long[] shapes = {1, dets.getRow(keep.getInt(i)).shape()[0]};
        dets2 = Nd4j.vstack(dets2,dets.getRow(keep.getInt(i)).reshape(shapes));
    }
    landnms = thresholdedLandnms.get(keep);

//     # keep top-K faster NMS
//     dets = dets[:args.keep_top_k, :]
//     landms = landms[:args.keep_top_k, :]

    System.out.println(Arrays.toString(keep.shape()));
    System.out.println(Arrays.toString(dets2.shape()));
    System.out.println(Arrays.toString(landnms.shape()));
    dets = Nd4j.hstack(dets2, landnms);

    System.exit(0);
   }
}
