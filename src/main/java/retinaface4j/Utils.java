package retinaface4j;

import java.util.*;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.util.ArrayUtil;
import org.nd4j.linalg.indexing.NDArrayIndex;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

public final class Utils {

    // Private constructor to prevent instantiation
    private Utils() {
        throw new UnsupportedOperationException();
    }

    public static INDArray substractMean(INDArray img, float[] pixelMeans){
        INDArray rMean = Nd4j.ones(1, 1, img.shape()[2], img.shape()[3]).mul(pixelMeans[0]);
        INDArray gMean = Nd4j.ones(1, 1, img.shape()[2], img.shape()[3]).mul(pixelMeans[1]);
        INDArray bMean = Nd4j.ones(1, 1, img.shape()[2], img.shape()[3]).mul(pixelMeans[2]);
        INDArray pixelMean = Nd4j.concat(1, rMean, gMean, bMean);
        return img.sub(pixelMean);
    }

    //public static methods here
    public static INDArray decode(float[] loc, float[] priors, long[] locShape){
        long[] locShapeSqueezed = {locShape[1], locShape[2]};
        long[] locShapeSqueezedSliced = {locShape[1], locShape[2]/2};

        INDArray nd4jLoc = Nd4j.createFromArray(loc).reshape(locShapeSqueezed);
        INDArray nd4jPriors = Nd4j.createFromArray(priors).reshape(locShapeSqueezed);

        INDArray locEnd = Nd4j.vstack(nd4jLoc.getColumn(2),nd4jLoc.getColumn(3)).transpose();
        INDArray locStart = Nd4j.vstack(nd4jLoc.getColumn(0),nd4jLoc.getColumn(1)).transpose();
        INDArray priorsEnd = Nd4j.vstack(nd4jPriors.getColumn(2),nd4jPriors.getColumn(3)).transpose();
        INDArray priorsStart = Nd4j.vstack(nd4jPriors.getColumn(0),nd4jPriors.getColumn(1)).transpose();

        INDArray stack1 = priorsStart.add(locStart.mul(0.1).mul(priorsEnd));
        INDArray stack2 = priorsEnd.mul(Transforms.exp(locEnd.mul(0.2)));

        INDArray stack1Trans = stack1.sub(stack2.mul(0.5));
        INDArray stack2Trans = stack2.add(stack1Trans);

        INDArray boxes = Nd4j.hstack(stack1Trans, stack2Trans);

        return boxes;
    }

    public static INDArray decode_landm(float[] landnms, float[] priors, long[] landnmsShape) {
        long[] landnmshapeSqueezed = {landnmsShape[1], landnmsShape[2]};
        long[] priorsShapeSqueezed = {landnmsShape[1], 4};
        long[] landnmsShapeSqueezedSliced = {landnmsShape[1], landnmsShape[2]/2};

        INDArray nd4jlandnms = Nd4j.createFromArray(landnms).reshape(landnmshapeSqueezed);
        INDArray nd4jPriors = Nd4j.createFromArray(priors).reshape(priorsShapeSqueezed);

        INDArray priorsEnd = Nd4j.vstack(nd4jPriors.getColumn(2),nd4jPriors.getColumn(3)).transpose();
        INDArray priorsStart = Nd4j.vstack(nd4jPriors.getColumn(0),nd4jPriors.getColumn(1)).transpose();

        INDArray landnms0 = Nd4j.vstack(nd4jlandnms.getColumn(0),nd4jlandnms.getColumn(1)).transpose();
        INDArray landnms1 = Nd4j.vstack(nd4jlandnms.getColumn(2),nd4jlandnms.getColumn(3)).transpose();
        INDArray landnms2 = Nd4j.vstack(nd4jlandnms.getColumn(4),nd4jlandnms.getColumn(5)).transpose();
        INDArray landnms3 = Nd4j.vstack(nd4jlandnms.getColumn(6),nd4jlandnms.getColumn(7)).transpose();
        INDArray landnms4 = Nd4j.vstack(nd4jlandnms.getColumn(8),nd4jlandnms.getColumn(9)).transpose();

        INDArray stack0 = priorsStart.add(landnms0.mul(0.1).mul(priorsEnd));
        INDArray stack1 = priorsStart.add(landnms1.mul(0.1).mul(priorsEnd));
        INDArray stack2 = priorsStart.add(landnms2.mul(0.1).mul(priorsEnd));
        INDArray stack3 = priorsStart.add(landnms3.mul(0.1).mul(priorsEnd));
        INDArray stack4 = priorsStart.add(landnms4.mul(0.1).mul(priorsEnd));

        INDArray output = Nd4j.hstack(stack0,stack1,stack2,stack3,stack4);

        return output;
    }


    public static INDArray cpu_nms(INDArray dets, double thresh) {
        INDArray x1 = dets.getColumn(0);
        INDArray y1 = dets.getColumn(1);
        INDArray x2 = dets.getColumn(2);
        INDArray y2 = dets.getColumn(3);
        INDArray scores = dets.getColumn(4);

        INDArray areas = x2.sub(x1).add(1).mul(y2.sub(y1).add(1));
        INDArray order = Nd4j.reverse(Nd4j.createFromArray(ArrayUtils.argsort(scores.toFloatVector())));

        int i = 0;
        boolean keepon = true;
        List<Integer> keep = new ArrayList<Integer>();
        while (keepon){
            i = order.getInt(0);
            keep.add(i);
            INDArray x1OrderInd = x1.get(order.get(NDArrayIndex.interval(1, order.shape()[0])));
            INDArray xx1 = Transforms.max(x1OrderInd,
                                          Nd4j.ones(x1OrderInd.shape()).mul(x1.getDouble(i)));
            INDArray y1OrderInd = y1.get(order.get(NDArrayIndex.interval(1, order.shape()[0])));
            INDArray yy1 = Transforms.max(y1OrderInd,
                                          Nd4j.ones(y1OrderInd.shape()).mul(y1.getDouble(i)));
            INDArray x2OrderInd = x2.get(order.get(NDArrayIndex.interval(1, order.shape()[0])));
            INDArray xx2 = Transforms.min(x2OrderInd,
                                          Nd4j.ones(x2OrderInd.shape()).mul(x2.getDouble(i)));

            INDArray y2OrderInd = y2.get(order.get(NDArrayIndex.interval(1, order.shape()[0])));
            INDArray yy2 = Transforms.min(y2OrderInd,
                                          Nd4j.ones(y2OrderInd.shape()).mul(y2.getDouble(i)));


            INDArray w = Transforms.max(Nd4j.zeros(xx2.shape()), xx2.sub(xx1).add(1));
            INDArray h = Transforms.max(Nd4j.zeros(yy2.shape()), yy2.sub(yy1).add(1));

            INDArray inter = w.mul(h);
            INDArray over = inter.div(areas.get(order.get(NDArrayIndex.interval(1, order.shape()[0]))).sub(inter).add(areas.getDouble(i)));
            List<Integer> indices = new ArrayList<Integer>();
            for (int j =0; j<over.shape()[0]; j++){
                if (over.getDouble(j)<=thresh){
                    indices.add(j);
                }
            }

            int[] indicesArr = new int[indices.size()];
            for (int j=0; j < indicesArr.length; j++)
            {
                indicesArr[j] = indices.get(j).intValue();
            }
            INDArray nd4jInds = Nd4j.createFromArray(indicesArr);
            if (nd4jInds.isEmpty()) {
            keepon = false;
            continue;
            }

            order = order.get(nd4jInds.add(1));
            if (order.isEmpty()) {
            keepon = false;
            }

        }
        int[] keepArr = new int[keep.size()];
        for (int j=0; j < keepArr.length; j++)
        {
            keepArr[j] = keep.get(j).intValue();
        }

        INDArray nd4jKeepArr = Nd4j.createFromArray(keepArr);
        return nd4jKeepArr;
    }
}