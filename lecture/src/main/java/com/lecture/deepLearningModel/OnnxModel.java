package com.lecture.deepLearningModel;

import ai.onnxruntime.OrtEnvironment;
import ai.onnxruntime.OrtSession;

/**
 * ONNX 모델 (하이브리드 모델) 
 */
public class OnnxModel {
	
	 public OrtEnvironment env;
     public OrtSession session;
     public long input;
     public long channels;
     public long netHeight;
     public long netWidth;
     public float confThreshold;
     public float nmsThreshold;

     public OnnxModel(OrtEnvironment env, OrtSession session, long input, long channels, long netHeight, long netWidth, float confThreshold, float nmsThreshold) {
         this.env = env;
         this.session = session;
         this.input = input;
         this.channels = channels;
         this.netHeight = netHeight;
         this.netWidth = netWidth;
         this.confThreshold = confThreshold;
         this.nmsThreshold = nmsThreshold;
     }
}
