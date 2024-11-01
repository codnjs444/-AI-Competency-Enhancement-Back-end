package com.lecture.hscode;

import java.nio.file.Path;
import java.nio.file.Paths;

import com.lecture.hscode.dataSet.HsCodeDataset;

import ai.djl.Model;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.training.listener.TrainingListener;

/*  
 * HsCode 학습
 */
public class HsCodeLrn {

	private static String modelNm = "kowon_hscode";
	private static Path modelDir = Paths.get("C:/YOLO_DATA/DJL_MODEL/");
	
	public static void main(String[] args) throws Exception {
		HsCodeLrn hsCodeLrn = new HsCodeLrn();
		hsCodeLrn.run();
	}
	 
	/*HsCode 학습 */
	public void run() throws Exception{
		System.out.println("학습시작");
		
		int batchSize = 20; //Gradient Descent를 한번 계산하기 위한 학습 데이터의 개수 (데이터 벡터 만들때 사용해도됨)
		int epochs = 5000; //전체 학습 데이터에 대해서 steps를 진행

		NDManager manager = NDManager.newBaseManager();
		 
		//hsCode dataset
		HsCodeDataset dataset = new HsCodeDataset.Builder()
		        .setManager(manager)
		        .setSampling(batchSize, false)
		        .build();
		dataset.prepare();//데이터 셋 사용
		
		//data , label 전처리 했따. 자연어에서 -> 임베딩작업 PC가 알아 볼 수 있도록 처리 했다.
        
		System.out.println("############# getDataChk:"+dataset.getDataChk("KNIT_SWEATSHIRTS/COTTON").get("value"));
		System.out.println("############# getDataChk:"+dataset.getDataChk("KNIT_SWEATSHIRTS/COTTON").get("key"));
		
		System.out.println("############# getLabelChk:"+dataset.getLabelChk("611020030E").get("value"));
		System.out.println("############# getLabelChk:"+dataset.getLabelChk("611020030E").get("key"));
		
		
		ArrayDataset trainIter = loadArray(dataset.getData(), dataset.getLabels(), batchSize, true);
		
	    //1. 모델 정의 여기부분 수정 더 해본다.
		SequentialBlock block = new SequentialBlock();
	    block.add(Blocks.batchFlattenBlock(dataset.getMaxToken()));
	    block.add(Linear.builder().setUnits(300).build());//히든레이어 
	    block.add(Activation::relu);
	    block.add(Linear.builder().setUnits(10).build()); //출력 레이어입니다.
	    block.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT); //초기화 설정
		
	    //훈련 및 예측
	    Tracker lrt = Tracker.fixed(0.01f); // 2.수정해본다. 
	    Optimizer adam = Optimizer.adam().optLearningRateTracker(lrt).build();

	    
	    // 3. KEYWORDS hscode 전처리 힌다. 
	    
	    Loss loss = Loss.l2Loss();
	    
		DefaultTrainingConfig config = new DefaultTrainingConfig(loss)
		                .optOptimizer(adam) // Optimizer (loss function)
		                .optDevices(Engine.getInstance().getDevices(1)) // single GPU or CPU
		                .addEvaluator(new Accuracy()) // Model 정확성
		                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging
		//MXNet PyTorch
		
		//MXNet epoch P50: 0.033 s, P90: 0.046 s Accuracy: 0.90, L2Loss: 0.09
		
		//PyTorch epoch P50: 0.015 s, P90: 0.020 s Accuracy: 0.96, L2Loss: 0.05
		try (Model model = Model.newInstance(modelNm , "MXNet")) {
			model.setBlock(block);
            
	        try (Trainer trainer = model.newTrainer(config)) {
	        	
	            trainer.initialize(new Shape(10, dataset.getMaxToken()));
	            trainer.setMetrics(new Metrics());
	            
	            EasyTrain.fit(trainer, epochs, trainIter, null);
	            
	            // 평가 결과 수집 
		        TrainingResult result = trainer.getTrainingResult();
		        
				model.setProperty("Epoch", String.valueOf(epochs));
		        model.setProperty("Accuracy", String.format("%.5f", result.getTrainEvaluation("Accuracy")));
		        model.setProperty("Loss", String.format("%.5f", result.getTrainLoss()));
	            
	            //학습 파일 저장
				model.save(modelDir, modelNm);
	        }
		    
		}
		
	}
	
	//정답,예측값
	public static ArrayDataset loadArray(NDArray features, NDArray labels, int batchSize, boolean shuffle) {
	    return new ArrayDataset.Builder()
	                  .setData(features) // 정답값 키값
	                  .optLabels(labels) // 예측값 몸무게
	                  .setSampling(batchSize, shuffle) // 배치 크기 및 무작위 샘플링 설정
	                  .build();
	}
	 
}
