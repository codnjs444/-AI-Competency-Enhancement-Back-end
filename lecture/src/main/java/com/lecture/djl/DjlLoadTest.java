package com.lecture.djl;


import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

public class DjlLoadTest {

	static String modelPath = "C:/YOLO_DATA/DJL_MODEL";
	public static final String model_name = "kokwon_ai";//본인AI학습 파일 

	
	public static void main(String[] args) throws Exception{
		DjlLoadTest djlLoadTest = new DjlLoadTest();
		djlLoadTest.loadRun();
	} 
	
	public void loadRun() throws Exception{
		//학습 파일을 실행 해 볼 예정입니다.
		float inputData [] = {2422,42235};//더하기 
		
		Path modelDir = Paths.get(modelPath);//학습파일 위치
		
		//모델 정의 중요 ****
		SequentialBlock block = new SequentialBlock();
		Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build(); // 아웃풋 몇개 
		block.add(linearBlock);
		
		// input , output
    	Criteria<float[], Float> translator =
                Criteria.builder()
                        .setTypes(float[].class, Float.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optModelName(model_name)//학습 파일 이름
                        .optTranslator(new MyTranslator())
                        .optProgress(new ProgressBar())
                        .build();
        
      	System.out.println("\n");
  		//System.out.println("translator :"+translator);
  		
  		//추론 (학습 데이터로 결과치를 보겠다)
  		try (ZooModel<float[], Float> model = translator.loadModel();
                Predictor<float[], Float> predictor = model.newPredictor()) {
  			//System.out.println("model :"+model.getName());
  			//System.out.println("model :"+model.getModelPath());
  			//System.out.println("model :"+model.getBlock());
  			
  			float predictResult = predictor.predict(inputData);//더하기 값 2개를 줘야 겠죠.
            System.out.println("AI 결과값 predictResult :"+predictResult);
  			
        }
	}
	
	//후처리 
	public static class  MyTranslator implements NoBatchifyTranslator<float[], Float> {
        MyTranslator() {}
        
        //input 
		@Override
		public NDList processInput(TranslatorContext ctx, float[] input) throws Exception {
			//입력 데이터 (전처리 할거 있으면 여기서  처리해.)
			NDArray inputIds = ctx.getNDManager().create(input , new Shape(1, 2)); //ndArray로 변환 
			
			//System.out.println("inputIds :"+inputIds);
			//System.out.println("inputIds.expandDims(0) :"+inputIds.expandDims(0));
	        return new NDList(inputIds.expandDims(0));
		}
		
		//output
		@Override
	    public Float processOutput(TranslatorContext ctx, NDList list) throws Exception {
	        NDArray opuput = list.get(0);
	        //System.out.println("opuput :"+opuput);
	       // System.out.println("opuput.getFloat(0) :"+opuput.getFloat(0));
	        float result = opuput.getFloat(0);
	        return result;
	    }
    }//MyTranslator


}
