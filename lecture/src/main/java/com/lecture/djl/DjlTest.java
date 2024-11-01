package com.lecture.djl;

import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;

import ai.djl.Model;
import ai.djl.engine.Engine;
import ai.djl.metric.Metrics;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDArrays;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
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

public class DjlTest {

	public static void main(String[] args) throws Exception {
		NDManager manager = NDManager.newBaseManager();
		
		DjlTest djlTest = new DjlTest();
		djlTest.aiRun(manager);
		
		//djlTest.NdArrayTest(manager);
		 
	}
	
	public void aiRun(NDManager manager) throws Exception {
		System.out.println("자 DJL 학습 한번 해보겠습니다.");
		
		String aiNm = "kokwon_ai"; //본인이름으로 
		int batchSize = 5;//전체 데이터에서 10씩 나눠서 학습한다.
		int epochs = 500;//학습 몇번 시킬거냐?? 
		
		//문제 (숫자 2자리 이하만 알려주시고 + 학습 해보겠습니다)
		NDArray data1 = manager.create(new float[] {1f, 2f},new Shape(1, 2));
		NDArray data2 = manager.create(new float[] {2f, 3f},new Shape(1, 2));
		NDArray data3 = manager.create(new float[] {3f, 4f},new Shape(1, 2));
		NDArray data4 = manager.create(new float[] {4f, 4f},new Shape(1, 2));
		NDArray data5 = manager.create(new float[] {6f, 2f},new Shape(1, 2));
		NDArray data6 = manager.create(new float[] {1f, 4f},new Shape(1, 2));
		NDArray data7 = manager.create(new float[] {3f, 5f},new Shape(1, 2));
		NDArray data8 = manager.create(new float[] {3f, 6f},new Shape(1, 2));
		NDArray data9 = manager.create(new float[] {1f, 7f},new Shape(1, 2));
		NDArray data10 = manager.create(new float[] {13f, 1f},new Shape(1, 2));
		NDArray data11 = manager.create(new float[] {4f, 2f},new Shape(1, 2));
		NDArray data12 = manager.create(new float[] {1f, 23f},new Shape(1, 2));
		NDArray data13 = manager.create(new float[] {5f, 34f},new Shape(1, 2));
		NDArray data14 = manager.create(new float[] {2f, 45f},new Shape(1, 2));
		NDArray data15 = manager.create(new float[] {2f, 6f},new Shape(1, 2));
		NDArray data16 = manager.create(new float[] {32f, 4f},new Shape(1, 2));
		NDArray data17 = manager.create(new float[] {51f, 3f},new Shape(1, 2));
		NDArray data18 = manager.create(new float[] {62f, 1f},new Shape(1, 2));
		NDArray data19 = manager.create(new float[] {13f, 2f},new Shape(1, 2));
		NDArray data20 = manager.create(new float[] {74f, 1f},new Shape(1, 2));
		NDArray data21 = manager.create(new float[] {31f, 0f},new Shape(1, 2));
		NDArray data22 = manager.create(new float[] {22f, 2f},new Shape(1, 2));
		NDArray data23 = manager.create(new float[] {43f, 2f},new Shape(1, 2));
		NDArray data24 = manager.create(new float[] {24f, 3f},new Shape(1, 2));
		NDArray data25 = manager.create(new float[] {15f, 1f},new Shape(1, 2));
		NDArray data26 = manager.create(new float[] {1f, 2f},new Shape(1, 2));
		NDArray data27 = manager.create(new float[] {222f, 4f},new Shape(1, 2));
		NDArray data28 = manager.create(new float[] {132f, 5f},new Shape(1, 2));
		NDArray data29 = manager.create(new float[] {511f, 2f},new Shape(1, 2));
		NDArray data30 = manager.create(new float[] {63f, 2f},new Shape(1, 2));

		NDList xNDList = new NDList();//여기 배열로 위에 데이터 30개를 넣었다.
		xNDList.add(data1);xNDList.add(data2);xNDList.add(data3);xNDList.add(data4);xNDList.add(data5);
		xNDList.add(data6);xNDList.add(data7);xNDList.add(data8);xNDList.add(data9);xNDList.add(data10);
		xNDList.add(data11);xNDList.add(data12);xNDList.add(data13);xNDList.add(data14);xNDList.add(data15);
		xNDList.add(data16);xNDList.add(data17);xNDList.add(data18);xNDList.add(data19);xNDList.add(data20);
		xNDList.add(data21);xNDList.add(data22);xNDList.add(data23);xNDList.add(data24);xNDList.add(data25);
		xNDList.add(data26);xNDList.add(data27);xNDList.add(data28);xNDList.add(data29);xNDList.add(data30);
		
		
		NDArray x = NDArrays.concat(xNDList).toType(DataType.FLOAT32, false);
		System.out.println("x :"+x);
		System.out.println("x :"+x.get(1));
		
		//답
		NDArray label1 = manager.create(new float[] {3f},new Shape(1, 1)); //  1 + 2 = 3
		NDArray label2 = manager.create(new float[] {5f},new Shape(1, 1));
		NDArray label3 = manager.create(new float[] {7f},new Shape(1, 1));
		NDArray label4 = manager.create(new float[] {8f},new Shape(1, 1));
		NDArray label5 = manager.create(new float[] {8f},new Shape(1, 1));
		NDArray label6 = manager.create(new float[] {5f},new Shape(1, 1));
		NDArray label7 = manager.create(new float[] {8f},new Shape(1, 1));
		NDArray label8 = manager.create(new float[] {9f},new Shape(1, 1));
		NDArray label9 = manager.create(new float[] {8f},new Shape(1, 1));
		NDArray label10 = manager.create(new float[] {14f},new Shape(1, 1));
		NDArray label11 = manager.create(new float[] {6f},new Shape(1, 1));
		NDArray label12 = manager.create(new float[] {24f},new Shape(1, 1));
		NDArray label13 = manager.create(new float[] {39f},new Shape(1, 1));
		NDArray label14 = manager.create(new float[] {47f},new Shape(1, 1));
		NDArray label15 = manager.create(new float[] {8f},new Shape(1, 1));
		NDArray label16 = manager.create(new float[] {36f},new Shape(1, 1));
		NDArray label17 = manager.create(new float[] {54f},new Shape(1, 1));
		NDArray label18 = manager.create(new float[] {63f},new Shape(1, 1));
		NDArray label19 = manager.create(new float[] {15f},new Shape(1, 1));
		NDArray label20 = manager.create(new float[] {75f},new Shape(1, 1));
		NDArray label21 = manager.create(new float[] {31f},new Shape(1, 1));
		NDArray label22 = manager.create(new float[] {24f},new Shape(1, 1));
		NDArray label23= manager.create(new float[] {45f},new Shape(1, 1));
		NDArray label24 = manager.create(new float[] {27f},new Shape(1, 1));
		NDArray label25 = manager.create(new float[] {16f},new Shape(1, 1));
		NDArray label26 = manager.create(new float[] {3f},new Shape(1, 1));
		NDArray label27 = manager.create(new float[] {226f},new Shape(1, 1));
		NDArray label28 = manager.create(new float[] {137f},new Shape(1, 1));
		NDArray label29 = manager.create(new float[] {513f},new Shape(1, 1));
		NDArray label30 = manager.create(new float[] {65f},new Shape(1, 1));

		NDList yNDList = new NDList();
		yNDList.add(label1);yNDList.add(label2);yNDList.add(label3);yNDList.add(label4);yNDList.add(label5);
		yNDList.add(label6);yNDList.add(label7);yNDList.add(label8);yNDList.add(label9);yNDList.add(label10);
		yNDList.add(label11);yNDList.add(label12);yNDList.add(label13);yNDList.add(label14);yNDList.add(label15);
		yNDList.add(label16);yNDList.add(label17);yNDList.add(label18);yNDList.add(label19);yNDList.add(label20);
		yNDList.add(label21);yNDList.add(label22);yNDList.add(label23);yNDList.add(label24);yNDList.add(label25);
		yNDList.add(label26);yNDList.add(label27);yNDList.add(label28);yNDList.add(label29);yNDList.add(label30);

		
		NDArray y = NDArrays.concat(yNDList).toType(DataType.FLOAT32, false);
		System.out.println("y :"+y);
		System.out.println("y :"+y.get(1));
		
		
		//import ai.djl.training.dataset.ArrayDataset;
		
		//x = 인풋 2개 (1+2) , y = 출력값 (3)
		ArrayDataset trainData = loadArray(x, y, batchSize, true);//데이터 셋 -> 질문(x) , 답(y)
		//data 1번값과 label 1번값을 뽑아보세요.
		System.out.println("data 10번값 :"+ trainData.get(manager, 10).getData().get(0) );
		System.out.println("label 10번값 :"+ trainData.get(manager,10).getLabels().get(0) );

		
		//모델 정의 
		SequentialBlock block = new SequentialBlock();
		Linear linearBlock = Linear.builder().optBias(true).setUnits(1).build(); // 아웃풋 몇개 
		block.add(linearBlock);
		
		
		//훈련 및 예측
		Tracker lrt = Tracker.fixed(0.1f); //훈련 몇프로 할거냐. 0.1f 쉬운값 위에 값처럼 , 0.0001  에러 계속 난다. 에러가 90%
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
	            trainer.initialize(new Shape(30,2)); // 에러 계속 난다. 에러가 90%
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
	
	
	//테스트 한 부분 
	public static void NdArrayTest(NDManager manager) {
		float[][] data = {{0.1f, 0.2f, 0.3f}, {0.4f, 0.5f, 0.6f}, {0.7f, 0.8f, 0.9f}};		
		
		NDArray x = manager.create(data);
		System.out.println("x:"+x);
		
		//현재값
		/*NDArray v1 = x.get(new NDIndex(":1"));  // 행 between:between , 열 
        System.out.println("v1 :"+v1);
		//2번째
		NDArray v2 = x.get(new NDIndex(":2, 0:1"));  // 행 between:between , 열 
		 //System.out.println("v2 :"+v2);
		 
		//3번값 x = 3*3
		// 0.5 , 0.6 
		 NDArray v3 = x.get(new NDIndex("1:2, 1:3"));  // 행 between:between , 열 
		 System.out.println("v3 :"+v3);
		 
		// 0.4f, 0.5f, 0.6f
		 NDArray v4 = x.get(new NDIndex("1:2, 1:3"));  // 행 between:between , 열 
		 System.out.println("여기를 0.4,0.5,0.6 나오도록 :"+v4);
		 
		 //0.1,0.2,0.3 뽑아보세요
		 NDArray v5 = x.get(new NDIndex("1:2, 1:3"));  // 행 between:between , 열 
		 System.out.println("v5 :"+v5);
		 */
		
		NDArray v6 = x.get(new NDIndex("1:2, 1:3"));  // 행 between:between , 열 
		System.out.println("0.8f, 0.9f 뽑아보세요. :"+v6);
	}
	

}

