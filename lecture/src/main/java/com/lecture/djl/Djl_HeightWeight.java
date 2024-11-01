package com.lecture.djl;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.Shape;
import ai.djl.nn.Activation;
import ai.djl.nn.SequentialBlock;
import ai.djl.nn.core.Linear;
import ai.djl.training.DefaultTrainingConfig;
import ai.djl.training.EasyTrain;
import ai.djl.training.Trainer;
import ai.djl.training.TrainingResult;
import ai.djl.training.dataset.ArrayDataset;
import ai.djl.training.evaluator.Accuracy;
import ai.djl.training.listener.TrainingListener;
import ai.djl.training.loss.Loss;
import ai.djl.training.optimizer.Optimizer;
import ai.djl.training.tracker.Tracker;
import tech.tablesaw.api.Table;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

public class Djl_HeightWeight {

	public static void main(String[] args) throws Exception {
		NDManager manager = NDManager.newBaseManager();
		
		Djl_HeightWeight heightWeight = new Djl_HeightWeight();
		heightWeight.aiRun(manager);
		 
	}
	
	public void aiRun(NDManager manager) throws Exception {
		System.out.println("자 DJL 학습 한번 해보겠습니다.");
		
		String aiNm = "kokwon_HeightWeight"; //본인이름으로 
		int batchSize = 10;//전체 데이터에서 10씩 나눠서 학습한다.
		int epochs = 3000;//학습 몇번 시킬거냐?? 
		
		
		Table col_height = csvNdArray("height");//엑셀에서 가져온 키(height) 데이터
		Table col_weight = csvNdArray("weight");//엑셀에서 가져온 몸무게(weight) 데이터
		
		System.out.println("col_height:"+col_height);
		System.out.println("col_weight:"+col_weight);

		//AI학습을 위한 벡터 list 만들기 
		NDArray x_array = manager.create(col_height.as().floatMatrix());// 키 데이터를 학습을 위해 NDArray로 변환
		NDArray y_array = manager.create(col_weight.as().floatMatrix());// 몸무게 데이터를 학습을 위해 NDArray로 변환

		System.out.println("키 데이터:"+x_array.get(0));
		System.out.println("몸무게 데이터:"+y_array.get(0));

		ArrayDataset trainData = loadArray(x_array, y_array, batchSize, false); // 키랑 몸무게 하나로 병합했다. 학습을 위해.
		
		//모델 정의 
		SequentialBlock block = new SequentialBlock();
		Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build(); // 아웃풋 몇개 
		block.add(Activation::relu);
		block.add(Linear.builder().setUnits(10).build());//히든레이어 
		block.add(linearBlock);
		
		
		//훈련 및 예측
		Tracker lrt = Tracker.fixed(0.005f); //훈련 몇프로 할거냐. 0.1f 쉬운값 위에 값처럼 , 0.0001  에러 계속 난다. 에러가 90%
		Optimizer adam = Optimizer.adam().optLearningRateTracker(lrt).build();//기울기 
		
		Loss loss = Loss.l2Loss();//로스율
		
		DefaultTrainingConfig config = new DefaultTrainingConfig(loss) //설정파일 
                .optOptimizer(adam) // Optimizer (loss function)
                .optDevices(Engine.getInstance().getDevices(1)) // single GPU or CPU
                .addEvaluator(new Accuracy()) // Model 정확성
                .addTrainingListeners(TrainingListener.Defaults.logging()); // Logging

		//학습 시작
		try (Model model = Model.newInstance(aiNm , "MXNet")) {//학습 시작
			model.setBlock(block); 
			
	        try (Trainer trainer = model.newTrainer(config)) {
	        	trainer.initialize(new Shape(1,1)); // 에러 계속 난다. 에러가 90%
	            trainer.setMetrics(new Metrics());
	            
	            System.out.println("## model :"+model.getBlock());	
	            System.out.println("## model :"+model.getName());
	            
	            EasyTrain.fit(trainer, epochs, trainData, null);//trainer 학습을 위해 모델 , epochs : 몇번학습 ,  trainIter 데이터(data,label)
	            
	            // 평가 결과 수집 
		        TrainingResult result = trainer.getTrainingResult();
				model.setProperty("Epoch", String.valueOf(epochs));
		        model.setProperty("Accuracy", String.format("%.5f", result.getTrainEvaluation("Accuracy")));
		        model.setProperty("Loss", String.format("%.5f", result.getTrainLoss()));
		        
		       //학습 파일 저장
				Path modelDir = Paths.get("C:/YOLO_DATA/DJL_MODEL/");
				Files.createDirectories(modelDir);
				model.save(modelDir, aiNm);
	        }
		}//학습 시작 end
		
		
	}
	
	//질문,답 (학습 데이터셋)
	public static ArrayDataset loadArray(NDArray x, NDArray y, int batchSize, boolean shuffle) {
	    return new ArrayDataset.Builder()
	                  .setData(x) // 정답값 2개 데이터  (1 + 2)
	                  .optLabels(y) // 예측값 더하기 정답값 (3)
	                  .setSampling(batchSize, shuffle) // 배치 크기 및 무작위 샘플링 설정
	                  .build();
	}	
	
	/*엑셀 csv 파일 데이터를 가져오는 메소드*/
	public static Table csvNdArray(String columns) {
		Table data = Table.read().file("C:/YOLO_DATA/Height_weight.csv");
		Table selectData = data.selectColumns(columns);//컬럼명 데이터 
		return selectData;
	}//csvNdArray

}

