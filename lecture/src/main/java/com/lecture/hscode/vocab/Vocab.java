package com.lecture.hscode.vocab;

import java.util.ArrayList;
import java.util.Collections;
import java.util.HashMap;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;

/**
 * 단어 
 */
public class Vocab {

    public int unk;
    public List<Map.Entry<String, Integer>> tokenFreqs;
    public List<String> idxToToken;
    public HashMap<String, Integer> tokenToIdx;

    public Vocab(String[][] tokens, int minFreq, String[] reservedTokens) {
        //빈도에 따라 정렬
        LinkedHashMap<?, Integer> counterObject = countCorpus2D(tokens);
        LinkedHashMap<String, Integer> counter = new LinkedHashMap<>();
        for (Map.Entry<?, Integer> e : counterObject.entrySet()) {
            counter.put((String) e.getKey(), e.getValue());
        }

        this.tokenFreqs = new ArrayList<>(counter.entrySet());
        tokenFreqs.sort((o1, o2) -> (o2.getValue()).compareTo(o1.getValue()));

        // 알 수 없는 토큰의 인덱스는 0으로 처리 
        this.unk = 0;
        List<String> uniqTokens = new ArrayList<>();
        uniqTokens.add("<unk>");
        Collections.addAll(uniqTokens, reservedTokens);
        for (Map.Entry<String, Integer> entry : tokenFreqs) {
            if (entry.getValue() >= minFreq && !uniqTokens.contains(entry.getKey())) {
                uniqTokens.add(entry.getKey());
            }
        }

        this.idxToToken = new ArrayList<>();
        this.tokenToIdx = new HashMap<>();
        for (String token : uniqTokens) {
            idxToToken.add(token);
            tokenToIdx.put(token, idxToToken.size() - 1);
        }
    }

    public int length() {
        return this.idxToToken.size();
    }

    public Integer[] getIdxs(String[] tokens) {
        List<Integer> idxs = new ArrayList<>();
        for (String token : tokens) {
            idxs.add(getIdx(token));
        }
        return idxs.toArray(new Integer[0]);
    }

    public Integer getIdx(String token) {
        return this.tokenToIdx.getOrDefault(token, this.unk);
    }

    public List<String> toTokens(List<Integer> indices) {
        List<String> tokens = new ArrayList<>();
        for (Integer index : indices) {
            tokens.add(toToken(index));
        }
        return tokens;
    }

    public String toToken(Integer index) {
        return this.idxToToken.get(index);
    }

    /** 토큰 빈도 계산. */
    public static <T> LinkedHashMap<T, Integer> countCorpus(List<T> tokens) {
        LinkedHashMap<T, Integer> counter = new LinkedHashMap<>();
        for (T token : tokens) {
            counter.put(token, counter.getOrDefault(token, 0) + 1);
        }
        return counter;
    }

    /** 토큰 목록을 토큰 목록으로 평면화 */
    public static <T> LinkedHashMap<T, Integer> countCorpus2D(T[][] tokens) {
        List<T> allTokens = new ArrayList<>();
        for (T[] token : tokens) {
            for (T t : token) {
                if (t != "") {
                    allTokens.add(t);
                }
            }
        }
        return countCorpus(allTokens);
    }
}