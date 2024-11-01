package com.lecture.hscode;


import java.io.FileReader;
import java.nio.file.Path;
import java.nio.file.Paths;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import com.lecture.hscode.dataSet.HsCodeDataset;

import ai.djl.inference.Predictor;
import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.types.DataType;
import ai.djl.nn.Activation;
import ai.djl.nn.Blocks;
import ai.djl.nn.Parameter;
import ai.djl.nn.SequentialBlock; 
import ai.djl.nn.core.Linear;
import ai.djl.repository.zoo.Criteria;
import ai.djl.repository.zoo.ZooModel;
import ai.djl.training.initializer.NormalInitializer;
import ai.djl.training.util.ProgressBar;
import ai.djl.translate.NoBatchifyTranslator;
import ai.djl.translate.TranslatorContext;

import java.util.Arrays;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;
import org.json.simple.parser.JSONParser;
import  java.util.Collections;

/*  
 * HsCode
 */

public class HsCodeLrnLoad {
	private static Logger log = LoggerFactory.getLogger(HsCodeLrn.class);
	private static String modelNm = "kowon_hscode";
	private static Path modelDir = Paths.get("C:/YOLO_DATA/DJL_MODEL/");
	
	public static void main(String[] args) throws Exception {
		HsCodeLrnLoad hsCodeLrnLoad = new HsCodeLrnLoad();
		hsCodeLrnLoad.run();
	}
	 
	/*HsCode 학습파일 실행 */
	public void run() throws Exception{
		NDManager manager = NDManager.newBaseManager();
		int batchSize = 4;
		
		HsCodeDataset dataset = new HsCodeDataset.Builder()
		        .setManager(manager)
		        .setSampling(batchSize, false)
		        .build();
		dataset.prepare();//데이터 셋 사용
		
		dataset.saveToken();//토큰값 json으로 저장 할때 사용	
		
		String input = "KNIT_SWEATSHIRTS/COTTON"; //611020030E
		//input = "KNIT_HOODIE/COTTON"; //611020030E
		//input = "BACKPAC/COTTON";
		//input = "WOVEN-PANTS/VELVET";
		//input = "SCARF/POLYESTER";
		//input = "WAIST BAG/COTTON";
		//input = "WOVEN-BAG/WOOL";//없는값
		//input = "SKIN CARE COSMETIC - EYE PATCHY(4)";
		//input = "GLASSES - ACRYLIC";

		//1. 모델 정의 여기부분 수정 더 해본다.
		SequentialBlock block = new SequentialBlock();
	    block.add(Blocks.batchFlattenBlock(dataset.getMaxToken()));
	    block.add(Linear.builder().setUnits(300).build());//히든레이어 
	    block.add(Activation::relu);
	    block.add(Linear.builder().setUnits(10).build()); //출력 레이어입니다.
	    block.setInitializer(new NormalInitializer(), Parameter.Type.WEIGHT); //초기화 설정
	    
	    //블락 
	    Criteria<String, String[]> translator =
                Criteria.builder()
                        .setTypes(String.class, String[].class)
                        .optModelPath(modelDir)
                        .optBlock(block)
                        .optModelName(modelNm)
                        .optTranslator(new MyTranslator(dataset))
                        .optProgress(new ProgressBar())
                        .build();
        
      	System.out.println("\n");
  		System.out.println("translator :"+translator);
  		
  		try (ZooModel<String, String[]> model = translator.loadModel();
                Predictor<String, String[]> predictor = model.newPredictor()) {
  			System.out.println("model :"+model.getName());
  			System.out.println("model :"+model.getModelPath());
  			System.out.println("model :"+model.getBlock());
  			
  			//추론
  			String[] predictResult = predictor.predict(input);
  			
  			for(int i=0; i< predictResult.length; i++) {
  	  			System.out.println("AI 결과값 -> "+predictResult[i]);  				
  			}
        } 
		    
	}
	
	//후처리 부분
	public static class  MyTranslator implements NoBatchifyTranslator<String, String[]> {
		private static HsCodeDataset dataset;
		MyTranslator(HsCodeDataset dataset) {
			MyTranslator.dataset = dataset;
        }
        
		@Override
		public NDList processInput(TranslatorContext ctx, String input) throws Exception {
			log.info("############# input:"+input);
			
			// KNIT_SWEATSHIRTS/COTTON 물어본값.
			
			//log.info("############# getDataChk:"+dataset.getDataChk(input).get("value"));
			//log.info("############# getDataChk:"+dataset.getDataChk(input).get("key"));
			//String inputTemp = dataset.getDataChk(input).get("key").toString();
			//log.info("############# inputTemp:"+inputTemp);
			
			String inputTemp = getToken(input);
			System.out.println(input+" => 토큰화:"+inputTemp);
			String [] stringToInteger = inputTemp.split(",");
			float [] ndArrayChange = new float [dataset.getMaxToken()]; 
			
			for(int i=0; i< dataset.getMaxToken(); i++) {
				if(stringToInteger.length <= i  ) {
					ndArrayChange[i] = Math.round(0);
				}else {
					ndArrayChange[i] = Math.round(Float.valueOf(stringToInteger[i].trim()));
				}
			}
			
			NDArray inputIds = ctx.getNDManager().create(ndArrayChange);
	        return new NDList(inputIds.expandDims(0));
		}
		
		@Override
	    public String[] processOutput(TranslatorContext ctx, NDList list) throws Exception {
			//후 처리 작업
			
			float aiResult [] = list.singletonOrThrow().toFloatArray(); //결과값
			
			System.out.print("AI 추론값 :");
			for(int i=0; i< aiResult.length; i++) {
				System.out.print(aiResult[i] +",");
			}
			
			//WOVEN-SHIRTS/NYLON -> 토큰화:22,6,24,1,7,13,3,16,9,10,5,3,18,7,17,12,6,7
			//AI 추론값 :6.579673,3.920993,1.5415263,6.7478347,7.361026,1.0581901,3.5695286,4.331082,1.5148907,3.081597,라벨모양(사이즈):98
			
			//620640210E -> 토큰화:7,4,1,7,8,1,4,5,1,3
			
			
			
	        NDArray getLabels = dataset.getLabels().toType(DataType.INT32, false); //라벨 데이터

	        System.out.println("라벨모양(사이즈):" + getLabels.getShape().get(0));//라벨 모양
	        
	        String[][] tokensLabel = dataset.getTokensLabel();
	        
	        
	        int labelSize = (int) getLabels.getShape().get(0);
	        String labelArray [] = new String [labelSize];
	        
	        for(int i=0; i<labelSize; i++ ) {
	        	labelArray[i] = getLabels.get(i).toString();
	        	System.out.println("labelArray[i]  : " + labelArray[i].replaceAll(",","").replaceAll(" ", "") );
	        }
	        
	        //확률 순으로 보기 
	        String [] maxPercentage = new String [labelArray.length];
	        for(int i=0; i<labelArray.length; i++ ) {
	        	  
	        	int delSt = labelArray[i].indexOf("[");
	  	        int delEnd = labelArray[i].indexOf("]");
	  	        String tempLabelChk  = labelArray[i].substring((delSt+1),delEnd).replaceAll(" ", ""); //라벨 토큰값만 빼오는 전처리작업
                
	  	        String [] tempLabel = tempLabelChk.split(","); 

	  	        float totalPercentage = 0f;
	  	        for(int z=0; z < tempLabel.length; z ++) {
	  	        	String percentage = "";
	  	        	
	  	        	//7.2383943,5.2193465,4.680497,1.9631319,3.9966166,1.3383982,1.1387322,5.5690746,0.78807783,3.1190858
	  				//[7551411613]          KNIT_SWEATSHIRTS/COTTON  hscode :611020030E -> 
	  	        	//8414241113
	  	        	
	  	        	//확률계산법 라벨토큰값*10/ai편향값
	  	        	if(aiResult[z] > Integer.valueOf(tempLabel[z]) ) {
	  	        		percentage = String.format("%.2f", Integer.valueOf(tempLabel[z]) * 10 / aiResult[z]);//확률계산
	  	        	}else {
	  	        		percentage = String.format("%.2f", aiResult[z]* 10 / Integer.valueOf(tempLabel[z]) );//확률계산
	  	        	}
	  	        	if(Float.valueOf(percentage) > 10) {
	  	        		percentage = "0";
	  	        	}
	  	        	//System.out.println("토큰값 (tempLabel) :" + tempLabel[z] + " ai 값 :"+aiResult[z] + " percentage:"+percentage);
	  	        	totalPercentage += Float.valueOf(percentage);
	  	        	
	  	        }
	  	        String hscode = Arrays.toString(tokensLabel[i]).replaceAll(",", "");
	  	        hscode = hscode.substring(1,hscode.length()-1).replaceAll(" ", "").toUpperCase();
	  	        maxPercentage[i] = String.valueOf(totalPercentage) +","+ hscode+"," +tempLabelChk;
	        }

	        Arrays.sort(maxPercentage,Collections.reverseOrder());//내림차순으로 정렬
	        maxPercentage = Arrays.stream(maxPercentage).distinct().toArray(String[]::new); //중복제거(라벨명이 같은게 많기 때문)
	        
	        //ai 추론값가 라벨값 확인 잠시 로그찍음.
	        System.out.println("토큰 :"+maxPercentage[0]);
	        System.out.println("토큰 :"+maxPercentage[1]);
	        System.out.println("토큰 :"+maxPercentage[2]);
	        
	        //제일 확률높은3개만 보여주기 
	        String [] resut = new String[3];
	        resut[0] = "HSCODE [" +maxPercentage[0].split(",")[1] + "] 확률:"+ maxPercentage[0].split(",")[0]; 
	        resut[1] = "HSCODE [" +maxPercentage[1].split(",")[1] + "] 확률:"+ maxPercentage[1].split(",")[0];
	        resut[2] = "HSCODE [" +maxPercentage[2].split(",")[1] + "] 확률:"+ maxPercentage[2].split(",")[0];
	        
	        return resut;
	    }
		
		/*인풋 스트링-> 토큰값으로 변경*/
		public String getToken(String input) {
			String result = "";//토큰값으로 전달 
			
			try {
				FileReader reader = new FileReader("C:/YOLO_DATA/hscode.json");
				
				Object obj = null;
				JSONArray jsonObj = null;
				JSONParser jsonParser = new JSONParser();

				obj = jsonParser.parse(reader);
				jsonObj = (JSONArray) obj;
				reader.close();
				
				//json 데이터 불러 온다음 input값으로 셋팅 값이 없으면 0으로 세팅한다. 
				JSONObject dataToken = new JSONObject();
				dataToken = (JSONObject)jsonObj.get(0);
				
				String [] temp =  input.toLowerCase().split("");
				
				//토큰 값으로 넣기
				for(int i=0 ;i <temp.length; i++) {
					result +=dataToken.get(temp[i])+",";
				}
				
				if(result.length() > 0) {
					result = result.substring(0, result.length()-1);	
				}
				result = result.replaceAll("null","0"); //학습시 없는 값은 0으로 
				
				System.out.println(input+ " 입력값을 토큰값으로 바꾼다. result :"+result);
				
				//%%% 단어:[k, n, i, t, _, s, w, e, a, t, s, h, i, r, t, s, /, c, o, t, t, o, n]
				//%%% key:[15, 7, 9, 5, 26, 3, 22, 1, 4, 5, 3, 16, 9, 10, 5, 3, 18, 8, 6, 5, 5, 6, 7]
				
			} catch (Exception e) {
				e.printStackTrace();
			}
			return result;
		}
		
		
    }
  	
}
