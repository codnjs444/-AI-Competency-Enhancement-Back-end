package com.lecture.deepLearningModel;

import java.io.BufferedReader;
import java.io.FileReader;
import java.nio.FloatBuffer;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.LinkedList;
import java.util.List;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import ai.onnxruntime.OnnxTensor;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;

/**
 * ONNX 모델 + OPENCV 사용 유틸 
 */
public class OnnxModelUtil {
	
	/*
     * 입력 이미지에 대한 경계 상자와 해당 클래스 레이블을 예측
     *
     * @param OpenCV Mat 형식의 입력 이미지
     * @return 좌표, 신뢰 점수 및 클래스 레이블이 있는 각 경계 상자에 대한 ArrayList<Float>를 포함하는 예측 결과.
     */
    public List<ArrayList<Float>> predictor(Mat img, OnnxModel onnxModel) {

        // OnnxTensor 전처리
        try (OnnxTensor tensor = transferTensor(img, onnxModel)) {

            OrtSession.Result result = onnxModel.session.run(Collections.singletonMap("images", tensor));
            OnnxTensor res = (OnnxTensor) result.get(0);

            //shape=[1, 5, 4725] 1인풋,5아웃풋, 4725 사이즈
            
            float[][] data = ((float[][][]) res.getValue())[0];
            // 각 행의 길이는 6이며 중심점, 너비, 높이, 신뢰도 점수 및 클래스 레이블을 나타냅니다.
            // 처음 4개의 값을 먼저 입력하고 xywh 부분을 배열로 바꿉니다
            
            //행렬 A의 열벡터가 [2,3]
            //              [5,0]  
            //	            [1,1]
            // 전치(Transpose)하면 행벡터인 [2,5,1], [3,0,1]로 변경
            Float[][] transpositionData = new Float[data[0].length][6];
            for (int i = 0; i < 4; i++) {
                for (int j = 0; j < data[0].length; j++) {
                    transpositionData[j][i] = data[i][j];
                }
            }
            
            // 각 경계 상자에 대해 가장 높은 신뢰도 점수와 해당 클래스 인덱스를 저장함.
            // 다섯 번째 요소는 신뢰도 점수를 나타내고 여섯 번째 요소는 클래스 레이블을 나타냅니다.
            for (int i = 0; i < data[0].length; i++) {
                for (int j = 4; j < data.length; j++) {
                    //설정된 신뢰도 점수가 원본 데이터보다 작은 경우 클래스 레이블과 함께 원본 데이터로 바꿉니다.
                    if (transpositionData[i][4] == null || transpositionData[i][4] < data[j][i]) {
                        transpositionData[i][4] = data[j][i]; // 신뢰도 점수
                        transpositionData[i][5] = (float) (j - 4); // class label
                    }
                }
            }

            List<ArrayList<Float>> boxes = new ArrayList<>();
            // 예측에 사용되는 이미지의 크기가 조정되므로 반환된 좌표는 크기가 조정된 이미지를 기준으로 함.
            // 따라서 최종 좌표를 원래 축척으로 복원해야 함.
            float scaleW = (float) img.width() / onnxModel.netWidth;
            float scaleH = (float) img.height() / onnxModel.netHeight;
            // 신뢰 임계값을 적용하고, xywh를 xyxy로 변환하고, 크기가 조정된 좌표를 복원함.
            for (Float[] d : transpositionData) {
                // 신뢰 임계값 적용
                if (d[4] > onnxModel.confThreshold) {
                    // xywh to xyxy
                    d[0] = d[0] - d[2] / 2;
                    d[1] = d[1] - d[3] / 2;
                    d[2] = d[0] + d[2];
                    d[3] = d[1] + d[3];
                    // 크기가 조정된 좌표를 복원하여 원래 좌표를 얻습니다.
                    d[0] = d[0] * scaleW;
                    d[1] = d[1] * scaleH;
                    d[2] = d[2] * scaleW;
                    d[3] = d[3] * scaleH;
                    
                    
                    // 원본 이미지의 경계 상자 위치를 계산함.
                    ArrayList<Float> box = new ArrayList<>(Arrays.asList(d));
                    boxes.add(box);
                }
            }

            return NMS(onnxModel, boxes);
        } catch (OrtException e) {
            throw new RuntimeException(e);
        }
    }
    
    /*
     * 모델 입력을 위해 OpenCV Mat 개체를 OnnxTensor 개체로 변환함.
     *
     * @param 변환할 이미지 데이터가 포함된 소스 Mat 객체.
     * @return 크기가 조정되고 색상 공간이 조정된 이미지 데이터를 포함하는 변환된 OnnxTensor 객체.
     * @throws OrtException ORT 런타임에서 오류가 발생하면 발생함.
     */
    public OnnxTensor transferTensor(Mat img, OnnxModel onnxModel) throws OrtException {
        Mat dst = new Mat();
        // 모델의 크기에 맞게 이미지 크기를 조정하세요.
        Imgproc.resize(img, dst, new Size(onnxModel.netWidth, onnxModel.netHeight));
        // 모델이 RGB 이미지로 훈련되었으므로 이미지 색상 공간을 BGR에서 RGB로 변환함.
        Imgproc.cvtColor(dst, dst, Imgproc.COLOR_BGR2RGB);
        // 이미지 데이터 유형을 32비트 부동 소수점으로 변환하고 각 픽셀 값을 0과 1 사이로 정규화함.
        dst.convertTo(dst, CvType.CV_32FC3, 1. / 255);
        // 너비*높이*채널 형식으로 조정된 이미지 데이터를 저장하는 배열을 만듭니다.
        float[] whc = new float[Long.valueOf(onnxModel.channels).intValue() * Long.valueOf(onnxModel.netWidth).intValue() * Long.valueOf(onnxModel.netHeight).intValue()];
        // Mat 객체에서 이미지 데이터를 가져와 배열에 저장함.
        dst.get(0, 0, whc);
        // 모델 입력에 이 형식이 필요하므로 너비-높이-채널(WHC) 형식 데이터를 채널 너비-높이(CHW) 형식으로 변환함.
        float[] chw = whc2cwh(whc);

        // 모델의 입력으로 사용될 조정된 이미지 데이터를 사용하여 OnnxTensor 개체를 만듭니다.
        return OnnxTensor.createTensor(onnxModel.env, FloatBuffer.wrap(chw), new long[]{onnxModel.input, onnxModel.channels, onnxModel.netWidth, onnxModel.netHeight});
    }
    
    /*모델 입력에 이 형식이 필요하므로 너비-높이-채널(WHC) 형식 데이터를 채널 너비-높이(CHW) 형식으로 변환함.*/
    public float[] whc2cwh(float[] img) {
        float[] chw = new float[img.length];
        int j = 0;
        for (int ch = 0; ch < 3; ++ch) {
            for (int i = ch; i < img.length; i += 3) {
                chw[j] = img[i];
                j++;
            }
        }
        return chw;
    }
    
    /*
     * NMS(Non-Maximum Suppression)를 수행하여 겹치는 경계 상자를 필터링하고 신뢰도 점수가 가장 높은 상자를 유지함.
     *  IOU를 통해 (두 박스의 교집합 / 두 박스의 합집합) 신뢰 점수가 가장 높은 박스만 남겨둠 
     * @param model NMS 임계값을 포함하는 모델 객체.
     * @param 상자 각 상자가 6개의 부동 소수점 배열 목록([x1, y1, x2, y2, 점수, classIndex])으로 표시되는 경계 상자 목록.
     * @param x1,y1 박스 상단 왼족 시작 점 , x2, y2 박스 하단 오른쪽 끝 점 
     * @return NMS 이후 필터링된 경계 상자 목록
     */
    public List<ArrayList<Float>> NMS(OnnxModel model, List<ArrayList<Float>> boxes) {
        int[] indexs = new int[boxes.size()];
        Arrays.fill(indexs, 1); // 모든 요소가 1로 설정된 인덱스 배열을 초기화함. 이는 모든 상자가 처음에 유지됨을 나타냅니다.

        // 각 상자를 반복함
        for (int cur = 0; cur < boxes.size(); cur++) {
            // 현재 상자가 제거 대상으로 표시된 경우 건너뛰기
            if (indexs[cur] == 0) {
                continue;
            }
            ArrayList<Float> curMaxConf = boxes.get(cur); // 현재 상자는 해당 클래스에 대한 신뢰도가 가장 높은 상자를 나타냅니다.

            // 현재 상자 이후의 상자를 반복함.
            for (int i = cur + 1; i < boxes.size(); i++) {
                // 현재 상자(비교 중인)가 제거 대상으로 표시된 경우 건너뜁니다.
                if (indexs[i] == 0) {
                    continue;
                }
                float classIndex = boxes.get(i).get(5);

                // 두 상자가 모두 동일한 클래스에 속하는 경우 NMS 수행
                if (classIndex == curMaxConf.get(5)) {
                    // 두 상자의 좌표를 얻으세요
                    float x1 = curMaxConf.get(0);
                    float y1 = curMaxConf.get(1);
                    float x2 = curMaxConf.get(2);
                    float y2 = curMaxConf.get(3);
                    
                    float x3 = boxes.get(i).get(0);
                    float y3 = boxes.get(i).get(1);
                    float x4 = boxes.get(i).get(2);
                    float y4 = boxes.get(i).get(3);

                    // 상자가 겹치지 않으면 건너뛰기
                    if (x1 > x4 || x2 < x3 || y1 > y4 || y2 < y3) {
                        continue;
                    }

                    // 상자 사이의 교차 면적을 계산함.
                    float intersectionWidth = Math.max(x1, x3) - Math.min(x2, x4);
                    float intersectionHeight = Math.max(y1, y3) - Math.min(y2, y4);
                    float intersectionArea = Math.max(0, intersectionWidth * intersectionHeight);

                    // 상자의 결합 면적을 계산함.
                    float unionArea = (x2 - x1) * (y2 - y1) + (x4 - x3) * (y4 - y3) - intersectionArea;

                    // IoU(Intersection over Union) 계산
                    float iou = intersectionArea / unionArea;

                    // IoU가 임계값을 초과하는 경우 제거 상자를 표시함.
                    indexs[i] = iou > model.nmsThreshold ? 0 : 1;
                }
            }
        }

        // 최종값 박스값 
        List<ArrayList<Float>> resBoxes = new LinkedList<>();
        
        for (int index = 0; index < indexs.length; index++) {
            if (indexs[index] == 1) {
                resBoxes.add(boxes.get(index));
            }
        }
        
        return resBoxes;
    }
    
    /*라벨링명 가져오기*/
    public List<String> getSynset(Path synsetDir , String model_name) throws Exception {
    	
    	List<String> result = new ArrayList<String>(); 
    	
        try (BufferedReader br = new BufferedReader(new FileReader(synsetDir +"/"+ model_name +"_labelNm.txt"))) {
            String line;
            while ((line = br.readLine()) != null) {
            	result.add(line);
            }
        }
		return result;
    }
    
}
