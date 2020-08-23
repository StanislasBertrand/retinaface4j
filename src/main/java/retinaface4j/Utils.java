package retinaface4j;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.common.util.ArrayUtil;

import java.util.Arrays;
import java.util.Comparator;
import java.util.Random;

public final class Utils {

    // Private constructor to prevent instantiation
    private Utils() {
        throw new UnsupportedOperationException();
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

//     public static int[] argsort(final float[] a, final boolean ascending) {
//         Integer[] indexes = new Integer[a.length];
//         for (int i = 0; i < indexes.length; i++) {
//             indexes[i] = i;
//         }
//         Arrays.sort(indexes, new Comparator<Integer>() {
//             @Override
//             public int compare(final Integer i1, final Integer i2) {
//                 return (ascending ? 1 : -1) * Float.compare(a[i1], a[i2]);
//             }
//         });
//         return asArray(indexes);
//     }
//
//     public static <T extends Number> int[] asArray(final T... a) {
//         int[] b = new int[a.length];
//         for (int i = 0; i < b.length; i++) {
//             b[i] = a[i].intValue();
//         }
//         return b;
//     }

    public static int cpu_nms(INDArray dets, double thresh) {
        INDArray x1 = dets.getColumn(0);
        INDArray y1 = dets.getColumn(1);
        INDArray x2 = dets.getColumn(2);
        INDArray y2 = dets.getColumn(3);
        INDArray scores = dets.getColumn(4);

        INDArray areas = x2.sub(x1).add(1).mul(y2.sub(y1).add(1));
        INDArray order = Nd4j.reverse(Nd4j.createFromArray(ArrayUtils.argsort(scores.toFloatVector())));
        System.out.println((order));
        return 1;
    }
//     def py_cpu_nms(dets, thresh):
//         """Pure Python NMS baseline."""
//         x1 = dets[:, 0]
//         y1 = dets[:, 1]
//         x2 = dets[:, 2]
//         y2 = dets[:, 3]
//         scores = dets[:, 4]
//
//         areas = (x2 - x1 + 1) * (y2 - y1 + 1)
//         order = scores.argsort()[::-1]
//
//         keep = []
//         while order.size > 0:
//             i = order[0]
//             keep.append(i)
//             xx1 = np.maximum(x1[i], x1[order[1:]])
//             yy1 = np.maximum(y1[i], y1[order[1:]])
//             xx2 = np.minimum(x2[i], x2[order[1:]])
//             yy2 = np.minimum(y2[i], y2[order[1:]])
//
//             w = np.maximum(0.0, xx2 - xx1 + 1)
//             h = np.maximum(0.0, yy2 - yy1 + 1)
//             inter = w * h
//             ovr = inter / (areas[i] + areas[order[1:]] - inter)
//
//             inds = np.where(ovr <= thresh)[0]
//             order = order[inds + 1]
//
//         return keep

}