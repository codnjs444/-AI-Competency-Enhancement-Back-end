package com.lecture.aiData;

import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;
import org.springframework.web.multipart.MultipartHttpServletRequest;

import com.lecture.deepLearningModel.OnnxModel;
import com.lecture.deepLearningModel.OnnxModelUtil;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;

@RestController
public class AiRestController {
	
	private final Logger log = LoggerFactory.getLogger(this.getClass().getSimpleName());
	
	private static OnnxModel onnxModel;
    private static OrtEnvironment environment;
    public static final float conf_threshold = 0.6f;//최소 신뢰점수  
    
	
	@PostMapping(value = "/getAiData") 
	public Map<String, Object> getAiData(MultipartHttpServletRequest request) throws Exception{
	
		log.info("##getAiData 호출 됨##");
		
		Map<String, Object> jsonResult = new HashMap<String, Object>();//APP로 전달  Map값

		Map<String, MultipartFile> fileMap = request.getFileMap();
		
		String fileSavePath = "C:/YOLO_DATA/serverFile/";

		for(String key : fileMap.keySet()) {
			MultipartFile file = fileMap.get(key);
			
			try {
				
				log.info("file :"+file.getName());
				
				String aiChkFilePath = fileSavePath;
				// 파일을 저장할 경로를 Path 객체로 받는다.
				Path directory = Path.of(aiChkFilePath).toAbsolutePath().normalize() ;
				Path targetPath = directory.resolve("temp.png").normalize();
				
				//파일 저장
				file.transferTo(targetPath);
				String saveFile = targetPath.toString();
				log.info("saveFile :"+saveFile);
				
				List<String> resultList = new ArrayList<String>();
				
				resultList = aiChk(saveFile);
				
				for(int i=0; i<resultList.size(); i++) {
					log.info("resultList :"+resultList.get(i));
					
				}
				
				
				jsonResult.put("resultList", resultList);
				
				//classId:eyebrow,x :276 y:143 width:52 height: 17 확률:0.91664183
		        //classId:eyes,x :189 y:171 width:47 height: 19 확률:0.94456494
		        //classId:eyes,x :282 y:173 width:45 height: 17 확률:0.9348172
				
			} catch (Exception e) {
				jsonResult.put("resultMsg", "파일 업로드 오류:"+e.getMessage());
			}
		}

		return jsonResult;
	}
	
	
	 public List <String> aiChk(String saveFile) throws Exception {
		 List <String> result = new ArrayList<String>();
		 
	    
		String dll_path = "C:/YOLO_DATA/OPENCV/opencv_java490.dll";//openCV load file (windows)
		String so_path = "C:/YOLO_DATA/OPENCV/libopencv_java490.so";;//openCV load file (nix,리룩스,aix)
		String modelPath = "C:/YOLO_DATA/";//학습 Model 
		String onnxModelNm = "face.onnx";//모델이름
		String testImgPath = saveFile;//ai 테스트 이미지
		
		OnnxModelUtil onnxModelUtil = new OnnxModelUtil();//ONNX 모델 + OPENCV 사용 유틸
		
		
		List<String> classNmArray = new ArrayList<String>();//클래스 라벨명
		classNmArray = onnxModelUtil.getSynset(Paths.get(modelPath) , "face");//라벨링TXT위치,라벨명  
		//System.out.println("classNmArray:"+classNmArray.size());
		 
		String osName = System.getProperty("os.name").toLowerCase();
        System.out.println("osName:"+osName);
        
        if (osName.contains("windows")) {
            System.load(dll_path);
        } else if (osName.contains("nix") || osName.contains("nux") || osName.contains("aix")) {
            System.load(so_path);
        } else {
            throw new UnsupportedOperationException("Unsupported operating system: " + osName);
        }
        
        environment = OrtEnvironment.getEnvironment();//onnx 엔진
        //C:/YOLO_DATA/face.onnx
        onnxModel = load(modelPath+onnxModelNm, 640, 640); //onnx 모델 로드 , 학습 사이즈
        
		Mat mat = Imgcodecs.imread(testImgPath);
		//예측
		List<ArrayList<Float>> predictor = onnxModelUtil.predictor(mat, onnxModel);

		System.out.println("===predictor.size() :"+predictor.size());
		
		for (ArrayList<Float> b : predictor) {
        	int x = b.get(0).intValue();
        	int y =b.get(1).intValue();
        	int width = (int) (b.get(2) - b.get(0));
        	int height = (int) (b.get(3) - b.get(1));
        	
        	int classIdx = b.get(5).intValue();
        	System.out.println("classIdx:"+classIdx+",x :" +x + " y:"+y + " width:"+width + " height: "+height + " 확률:"+b.get(4) + " classId:"+classNmArray.get(b.get(5).intValue()));
        	
        	
        	String classNm = classNmArray.get(b.get(5).intValue());
        	
        	String tempString = classNm + "," +x+","+y+","+width+","+height+","+b.get(4) + "|"; 
        	result.add(tempString);//서버로 보낼 데이터 
        }
		
		if(predictor.size() > 0) {
			
		}else {
			System.out.println("AI 예측 결과가 없습니다.");
		}
		
		return result;
    }
	 
	/*
     * 지정된 경로에서 ONNX 모델을 로드하고 모델 로드
     */
    private static OnnxModel load(String path, long height, long width) throws OrtException {

        OrtSession session = environment.createSession(path, new OrtSession.SessionOptions());
        Map<String, NodeInfo> infoMap = session.getInputInfo();
        TensorInfo nodeInfo = (TensorInfo) infoMap.get("images").getInfo();

        long input = 1;
        long channels = nodeInfo.getShape()[1];
        long netHeight = height;
        long netWidth = width;
        float nmsThreshold = 0.5f;

        return new OnnxModel(environment, session, input, channels, netHeight, netWidth, conf_threshold, nmsThreshold);
    }
	    
}
