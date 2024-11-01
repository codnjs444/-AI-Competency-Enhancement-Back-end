package com.lecture.onnx;

import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;

import ai.onnxruntime.NodeInfo;
import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtException;
import ai.onnxruntime.OrtSession;
import ai.onnxruntime.TensorInfo;
import com.lecture.deepLearningModel.OnnxModel;
import com.lecture.deepLearningModel.OnnxModelUtil;

import javax.imageio.ImageIO;
import javax.swing.*;
import java.awt.*;
import java.awt.image.BufferedImage;
import java.io.File;
import java.nio.file.Paths;
import java.util.*;
import java.util.List;

/**
 * onnxruntime + opencv 추론
 */
public class TestMainOnnx {
	
	private static OnnxModel onnxModel;
    private static OrtEnvironment environment;
    public static final float conf_threshold = 0.6f;//최소 신뢰점수  
    List<String> classNmArray = new ArrayList<String>();//클래스 라벨명
    		
    OnnxModelUtil onnxModelUtil = new OnnxModelUtil();//ONNX 모델 + OPENCV 사용 유틸

    public static void main(String[] args) throws Exception {
    	TestMainOnnx testMainOnnx = new TestMainOnnx();
    	testMainOnnx.run();
    }
    
    public void run() throws Exception {
    	
		String dll_path = "C:/YOLO_DATA/OPENCV/opencv_java490.dll";//openCV load file (windows)
		String so_path = "C:/YOLO_DATA/OPENCV/libopencv_java490.so";;//openCV load file (nix,리룩스,aix)
		String modelPath = "C:/YOLO_DATA/";//학습 Model 
		String onnxModelNm = "face.onnx";//모델이름
		String testImgPath = "C:/YOLO_DATA/testImg/t2.png";//ai 테스트 이미지
		
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
        	
        }
		
		if(predictor.size() > 0) {
			// draw
			BufferedImage image = ImageIO.read(new File(testImgPath));
			BufferedImage out = drawImage(image, predictor, modelPath);
			displayImage(out,"ONNX테스트");
		}else {
			System.out.println("AI 예측 결과가 없습니다.");
		}
		
    }
    
    
    /**
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

    /**
     * 지정된 BufferedImage에 일련의 경계 상자를 그리고 클래스 및 신뢰도 점수로 각 상자에 레이블을 지정함.
     *
     * @param image 경계 상자가 그려질 BufferedImage.
     * @param 상자 각 경계 상자에 대한 위치 및 클래스 신뢰도 정보를 포함하는 ArrayList 목록.
     * 		  각 ArrayList에는 왼쪽 상단 x 좌표, y 좌표,
     * 	      경계 상자의 오른쪽 하단 x 좌표, y 좌표 및 클래스 신뢰도 점수.
     * @return 경계 상자와 레이블이 그려진 BufferedImage.
     */
    public BufferedImage drawImage(BufferedImage image, List<ArrayList<Float>> boxs , String modelPath) throws Exception{
        Graphics graphics = image.getGraphics();

        
        
        graphics.setFont(new Font("Arial", Font.BOLD, 15));
        for (ArrayList<Float> b : boxs) {
        	
        	int x = b.get(0).intValue();
        	int y =b.get(1).intValue();
        	int width = (int) (b.get(2) - b.get(0));
        	int height = (int) (b.get(3) - b.get(1));
        	
        	//박스
            graphics.setColor(Color.RED);
            graphics.drawRect( x, y, width, height);

    		System.out.println("classNmArray:"+classNmArray);
    		
            //클래스 정보 
            String classNm = classNmArray.get(b.get(5).intValue());
            
            double predictor = Math.round(b.get(4)*100);//추론값
            
            //클래스 ID 
            graphics.setColor(Color.BLUE);
            graphics.drawString(classNm+":"+ predictor, x, (y+height)+13);
        }
        graphics.dispose();
        return image;
    }

    //자바 스윙으로 GUI로 보기 
    public static void displayImage(Image img, String title) {
        ImageIcon icon = new ImageIcon(img);
        JFrame frame = new JFrame(title);
        frame.setLayout(new FlowLayout());
        frame.setSize(img.getWidth(null) + 50, img.getHeight(null) + 50);
        JLabel lbl = new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }

}
