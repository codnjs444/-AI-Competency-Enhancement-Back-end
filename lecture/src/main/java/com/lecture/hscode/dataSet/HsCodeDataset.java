package com.lecture.hscode.dataSet;


import java.io.FileWriter;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.json.simple.JSONArray;
import org.json.simple.JSONObject;

import com.lecture.hscode.vocab.Vocab;

import ai.djl.ndarray.NDArray;
import ai.djl.ndarray.NDList;
import ai.djl.ndarray.NDManager;
import ai.djl.ndarray.index.NDIndex;
import ai.djl.ndarray.types.DataType;
import ai.djl.ndarray.types.Shape;
import ai.djl.training.dataset.RandomAccessDataset;
import ai.djl.util.Pair;
import ai.djl.util.Progress;
import tech.tablesaw.api.Table;
import ai.djl.ndarray.NDArrays;
import ai.djl.training.dataset.Record;

/*
 * hsCode 데이터셋
 */
public class HsCodeDataset extends RandomAccessDataset {

	private Vocab vocab;
	private Vocab vocabLabels;
    private NDArray data;
    private NDArray labels;
    private NDManager manager;
    private boolean prepared;
    private int maxToken = 0;
    private String[][] tokensData;
    private String[][] tokensLabel;		
    
    Pair<List<Integer>, Vocab> corpusVocabData = null;
    Pair<List<Integer>, Vocab> corpusVocabLabel = null;

    public HsCodeDataset(Builder builder) {
        super(builder);
        this.manager = builder.manager;
        this.data = this.manager.create(new Shape(1,65), DataType.FLOAT32);
        this.labels = this.manager.create(new Shape(1,10), DataType.FLOAT32);
        this.prepared = false;
    }

    @Override
    public Record get(NDManager manager, long index) throws IOException {
        NDArray X = data.get(new NDIndex("{}", index));
        NDArray Y = labels.get(new NDIndex("{}", index));
        return new Record(new NDList(X), new NDList(Y));
    }

    @Override
    protected long availableSize() {
        return data.getShape().get(0);
    }

    @Override
    public void prepare(Progress progress) throws IOException{
    	//init 기본 정의
    	if (prepared) {
            return;
        }
    	
    	//실행 
        try {
        	//data , Label  = 전처리 한다. 
        	this.corpusVocabData = loadCorpus(282,"KEYWORDS","singly");//데이터 임베딩 하기 단어로 짜르기 (slash line space singly)
        	this.corpusVocabLabel = loadCorpus(97,"HSCODE","singly");//데이터 임베딩 하기  단어로 짜르기 (slash line space singly)
        	
        } catch (Exception e) {
            e.printStackTrace(); 
        }
        
        this.vocab = corpusVocabData.getValue();
        this.vocabLabels = corpusVocabLabel.getValue();
        
        
        //data NDList로 넣기
        NDList xNDList = new NDList(); 
		for(int i=0; i<tokensData.length; i++) {
			
			//KNIT_SWEATSHIRTT/COTTON
			System.out.println("%%% 단어:" + Arrays.toString(tokensData[i]));
			System.out.println("%%% key:" + Arrays.toString(vocab.getIdxs(tokensData[i])));
			
			NDArray dataTemp = manager.zeros(new Shape(1,maxToken), DataType.FLOAT32);
		    List<String> token = Arrays.asList(Arrays.toString(vocab.getIdxs(tokensData[i])).substring(1, Arrays.toString(vocab.getIdxs(tokensData[i])).length()-1).split(",\\s*"));
		    for(int x=0; x< token.size(); x++) {
		    	dataTemp.set(new NDIndex(0,x), Integer.valueOf(token.get(x)) );
		    }
		    xNDList.add(dataTemp);
		}
		//System.out.println("%%%xNDList size:" + xNDList.size());
		//System.out.println("%%%xNDList:" + xNDList.get(2));
		
		//%%% 단어:[k, n, i, t, _, s, w, e, a, t, s, h, i, r, t, s, /, c, o, t, t, o, n]
		//%%% key:[15, 7, 9, 5, 26, 3, 22, 1, 4, 5, 3, 16, 9, 10, 5, 3, 18, 8, 6, 5, 5, 6, 7]
		
        
		this.data = NDArrays.concat(xNDList).toType(DataType.FLOAT32, false); // 지도학습 값
		//System.out.println("%%% data :"+data);
		//System.out.println("%%% data :"+data.get(0));
		
		this.vocab = corpusVocabData.getValue();
        this.vocabLabels = corpusVocabLabel.getValue();
		
        //data 처리 끝
        
        //라벨 넣어야죠.
		//라벨 NDList로 넣기
		NDList yNDList = new NDList(); //예측값
		for(int i=0; i<tokensLabel.length; i++) {
			NDArray labelTemp = manager.zeros(new Shape(1,10), DataType.FLOAT32);
		    List<String> token = Arrays.asList(Arrays.toString(vocabLabels.getIdxs(tokensLabel[i])).substring(1, Arrays.toString(vocabLabels.getIdxs(tokensLabel[i])).length()-1).split(",\\s*"));
		    for(int x=0; x< token.size(); x++) {
		    	labelTemp.set(new NDIndex(0,x), Integer.valueOf(token.get(x)) );
		    }
		    
		    //KNIT_SWEATSHIRTS/COTTON -> hscode	611020030E
		    
		    yNDList.add(labelTemp);
		}
		
		this.labels = NDArrays.concat(yNDList).toType(DataType.FLOAT32, false); // 예상 값
		//System.out.println("%%% labels :"+labels);
		//System.out.println("%%% labels :"+labels.get(0));
        this.prepared = true;
        
    }

    public Vocab getVocab() {
        return this.vocab;
    }
    
    public Vocab getVocabLabel() {
        return this.vocabLabels;
    }
    
    public int getMaxToken() {
        return this.maxToken;
    }
    
    public NDArray getData() {
        return this.data;
    }
    
    public NDArray getLabels() {
        return this.labels;
    }
    
    public String[][] getTokensLabel() {
        return this.tokensLabel;
    }
    
    //get 임베딩 데이터 확인 
    public Map<String, Object> getDataChk(String input)  {
    	Map<String, Object> resultMap = new HashMap<>(); 
    	input = input.toLowerCase();
    	int tokenNo = 0;
    	try {
    		String[] lines = StringData("KEYWORDS");

    		System.out.println("[data] lines:"+lines.length);
    		
    		for(int i=0; i<lines.length;i++) {
    			
    			if(input.equals(lines[i])) {
    				tokenNo = i;
    				System.out.println("[data] tokenNo:"+tokenNo);
    			}
    		}
    		String[][] tokens = tokenize(lines, "singly"); 
    		
    		String value = Arrays.toString(tokens[tokenNo]);
    		String key = Arrays.toString(vocab.getIdxs(tokens[tokenNo]));
    		
    		resultMap.put("value", value);
    		resultMap.put("key", key);
    		
    		
    	}catch (Exception e) {
    		System.out.println("getDataChk error :"+e);
		}
        return resultMap;
    }
    
    //get 임베딩 라벨 확인 
    public Map<String, Object> getLabelChk(String input)  {
    	Map<String, Object> resultMap = new HashMap<>(); 
    	input = input.toLowerCase();
    	int tokenNo = 0;
    	try {
    		String[] lines = StringData("HSCODE");

    		for(int i=0; i<lines.length;i++) {
    			if(input.equals(lines[i])) {
    				tokenNo = i;
    			}
    		}
    		String[][] tokens = tokenize(lines, "line");
    		
    		String value = Arrays.toString(tokens[tokenNo]);
    		String key = Arrays.toString(vocabLabels.getIdxs(tokens[tokenNo]));
    		
    		resultMap.put("value", value);
    		resultMap.put("key", key);
    		
    	}catch (Exception e) {
    		System.out.println("getDataChk error :"+e);
		}
        return resultMap;
    }
    
    
    public static final class Builder extends BaseBuilder<Builder> {
        int numSteps;
        NDManager manager;

        @Override
        protected Builder self() { return this; }

        public Builder setSteps(int steps) {
            this.numSteps = steps;
            return this;
        }

        public Builder setManager(NDManager manager) {
            this.manager = manager;
            return this;
        }

        public HsCodeDataset build() throws Exception{
        	HsCodeDataset dataset = new HsCodeDataset(this);
            return dataset;
        }
    }
    
    /*말뭉치 만들기*/
    public Pair<List<Integer>, Vocab> loadCorpus(int maxTokens, String column, String token) throws IOException, Exception {
    	
    	System.out.println("$$$$$$$ loadCorpus :"+column);
    	
    	/*
    	 * 가장 많이 쓰는것들이 뒤로 가고 
    	 * 적게 쓰는 S , / 앞으로 가라 
    	 */
	    /* 토큰 인덱스와 데이터 세트의 단어를 반환합니다.*/
    	String[] lines = StringData(column);
    	
	    String[][] tokens = tokenize(lines, token);//slash line space singly
	    Vocab vocab = new Vocab(tokens, 0, new String[0]);
	    // 데이터세트의 각 텍스트 줄은 반드시
	    // 문장 또는 단락, 모든 텍스트 줄을 단일 목록으로 병합
	    System.out.println("[data]:maxTokens :"+maxTokens);
	    System.out.println("[data]:tokens.length :"+tokens.length + " column:"+column);
	    
	    List<Integer> corpus = new ArrayList<>();
	    for (int i = 0; i < tokens.length; i++) {
	        for (int j = 0; j < tokens[i].length; j++) {
	            if (tokens[i][j] != "") {
	            	
	                corpus.add(vocab.getIdx(tokens[i][j]));
	            }
	        }
	    }
	    
	    if (maxTokens > 0) {
	    	corpus = corpus.subList(0, maxTokens);
	    }

	    //KEYWORDS에서 가장 큰 토큰값 찾기
	    int getMaxToken=0;
	    if(column.equals("KEYWORDS")) {
	    	for(int i=0; i<tokens.length; i++) {
				if(getMaxToken < tokens[i].length) {
					getMaxToken = tokens[i].length;
				}
			}
	 		this.maxToken = getMaxToken;
	 		this.tokensData = tokens;
	    }else if(column.equals("HSCODE")) {
	    	this.tokensLabel = tokens;
	    }
	    System.out.println("getMaxToken :"+getMaxToken);
	    
	    
	    return new Pair<List<Integer>, Vocab>(corpus, vocab);
	}
    
    //인코딩 작업 - 문자배열로 변환 소문자로 전체 치환
  	public String[] StringData(String csvNm){
  		Table col_keywords = csvNdArray(csvNm);
  		int size = col_keywords.rowCount();
  		String[] lines = new String[size];
  		for(int i=0;i<size; i++) {
  			lines[i] = (String)col_keywords.get(i, 0);
  			lines[i] = lines[i].toLowerCase();
  		}
  		
  	    return lines;
  	}
  	
  	//토큰화 시키기
  	public String[][] tokenize(String[] lines, String token) throws Exception {
  		
  	    //텍스트 줄을 단어 또는 문자 토큰으로 분할합니다. 
  	    String[][] output = new String[lines.length][];
  	    if (token.equals("slash")) {
  	        for (int i = 0; i < output.length; i++) {
  	            output[i] = lines[i].split("/");
  	        }
  	    }else if (token.equals("line")) {
  	    	//line
  	    	for (int i = 0; i < output.length; i++) {
  	    		output[i] =  lines[i].split("\n");
  	  	    }
  	    }else if (token.equals("space")) {
            for (int i = 0; i < output.length; i++) {
                output[i] = lines[i].split(" ");
            }
        }else if (token.equals("singly")) {
            for (int i = 0; i < output.length; i++) {
                output[i] = lines[i].split("");
            }
        }else {
  	        throw new Exception("ERROR: unknown token type: " + token);
  	    }
  	  
  	    return output;
  	}
  	 
  	@SuppressWarnings("unchecked")
	public void saveToken() throws Exception {
  		String[] lines = StringData("KEYWORDS");
    	
	    String[][] tokens = tokenize(lines, "singly");//slash line space singly
	    Vocab vocab = new Vocab(tokens, 0, new String[0]);
	    
	    JSONObject data = new JSONObject();//json data 파일에 저장
	    
	    for (int i = 0; i < tokens.length; i++) {
	        for (int j = 0; j < tokens[i].length; j++) {
	            if (tokens[i][j] != "") {
	            	data.put(tokens[i][j], String.valueOf(vocab.getIdx(tokens[i][j])));
	            }
	        }
	    }
  	
  		System.out.println("data:"+data);
  		
  		lines = StringData("HSCODE");
    	
	    tokens = tokenize(lines, "singly");//slash line space singly
	    vocab = new Vocab(tokens, 0, new String[0]);
	    
	    JSONObject label = new JSONObject();//json label 파일에 저장
	    
	    for (int i = 0; i < tokens.length; i++) {
	        for (int j = 0; j < tokens[i].length; j++) {
	            if (tokens[i][j] != "") {
	            	label.put(tokens[i][j], String.valueOf(vocab.getIdx(tokens[i][j])));
	            }
	        }
	    }
	    System.out.println("label:"+label);
  		
	    JSONArray jsonData = new JSONArray();
	    jsonData.add(data);
	    jsonData.add(label);
	    
	    System.out.println("jsonData:"+jsonData);
	    
		try {
			FileWriter file = new FileWriter("C:/YOLO_DATA/hscode.json");
			file.write(jsonData.toJSONString());
			file.flush();
			file.close();
		} catch (IOException e) {
			e.printStackTrace();
		}

  	}
  	
  	//데이터 가져오기 
  	public Table csvNdArray(String columns) {
  		Table data = Table.read().file("C:/YOLO_DATA/hscode_change.csv");
  		Table selectData = data.selectColumns(columns);//컬럼명 데이터 
  		return selectData;
  	}
}
