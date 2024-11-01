package com.lecture.djl;


import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.nn.Activation;
import ai.djl.nn.ParameterList;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

public class Djl_LoadHeightWeight {

	static String modelPath = "C:/YOLO_DATA/DJL_MODEL";
	public static final String model_name = "kokwon_HeightWeight";//본인AI학습 파일 

	
	public static void main(String[] args) throws Exception{
		Djl_LoadHeightWeight loadHeightWeight = new Djl_LoadHeightWeight();
		loadHeightWeight.loadRun();
	} 
	
	public void loadRun() throws Exception{
		
		//학습 파일을 실행 해 볼 예정입니다.
		float inputData = 173.6f;//키 데이터 
				
		Path modelDir = Paths.get(modelPath);//학습파일 위치
		
		//모델 정의 
		SequentialBlock block = new SequentialBlock();
		Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build(); // 아웃풋 몇개 
		block.add(Activation::relu);
		block.add(Linear.builder().setUnits(1).build());//히든레이어 
		block.add(linearBlock);
		
		// input 1개 , output 1개
    	Criteria<Float, Float> translator =
                Criteria.builder()
                        .setTypes(Float.class, Float.class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optModelName(model_name)
                        .optTranslator(new MyTranslator())
                        .optProgress(new ProgressBar())
                        .build();

        
      	System.out.println("\n");
  		//System.out.println("translator :"+translator);
  		
  		//추론 (학습 데이터로 결과치를 보겠다)
  		try (ZooModel<Float, Float> model = translator.loadModel();
                Predictor<Float, Float> predictor = model.newPredictor()) {
  			System.out.println("model :"+model.getName());
  			System.out.println("model :"+model.getModelPath());
  			System.out.println("model :"+model.getBlock());
  			
  			//확인방법
	        ParameterList params = block.getParameters();
	        NDArray wParam = params.valueAt(0).getArray();
			NDArray bParam = params.valueAt(1).getArray();
			
			System.out.println("wParam:"+wParam); //가중치 
			System.out.println("bParam:"+bParam);//편향
			
			//키 :191, 무게:	79

			//답 = 키 * 가중치 (w) - bParam;
			float ai_result = (inputData * wParam.getFloat(0)) + bParam.getFloat(0);
			
			
			System.out.println("ai_result:"+ai_result);//편향
  			
  			
  			float predictResult = predictor.predict(inputData);//더하기 값 2개를 줘야 겠죠.
            System.out.println("입력데이터:키 "+inputData+", AI 예측 몸무게:"+predictResult);
  		}

	}
	
  	//후처리 
	public static class  MyTranslator implements NoBatchifyTranslator<Float, Float> {
        MyTranslator() {}
        
        //input 
		@Override
		public NDList processInput(TranslatorContext ctx,  Float input) throws Exception {
			//입력 데이터 (전처리 할거 있으면 여기서  처리해.)
			//float inputData = 158.9; 키
			NDArray inputIds = ctx.getNDManager().create(input); //ndArray로 변환 
	        return new NDList(inputIds.expandDims(0));
		}
		
		//output
		@Override
	    public Float processOutput(TranslatorContext ctx, NDList list) throws Exception {
	        NDArray opuput = list.get(0);
	        System.out.println("opuput :"+opuput);
	        System.out.println("opuput.getFloat(0) :"+opuput.getFloat(0));
	        float result = opuput.getFloat(0);
	        return result;
	    }
    }//MyTranslator

}
