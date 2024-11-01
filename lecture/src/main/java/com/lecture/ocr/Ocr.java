package com.lecture.ocr;
import net.sourceforge.tess4j.ITesseract;
import net.sourceforge.tess4j.Tesseract;
import java.io.File;

public class Ocr {

	public static void main(String[] args) {
		System.out.println("OCR 시작");
	      try{
	         //코딩 합니다.
	         ITesseract instance = new Tesseract(); // 라이브러리 사용을 위한 선언
	         instance.setDatapath("C:/tessdata"); // 학습 파일 들어 있는 폴더  
	         instance.setLanguage("eng+kor");

	         //테스트 할 이미지
	         File imgPath = new File("C:/tessdata/testImg/eng_kor_num.png");
	         
	         String result = instance.doOCR(imgPath);
	         System.out.println("결과값 :"+result);
	         
	      }catch (Exception e) {
	         System.out.println("OCR ERROR :"+e.getMessage());
	      }
	      
	}
	
	
	
	
	
	

}
