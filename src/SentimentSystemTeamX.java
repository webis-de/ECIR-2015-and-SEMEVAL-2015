import java.io.BufferedReader;
import java.io.File;
import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.io.StringReader;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.Iterator;
import java.util.List;
import java.util.Map;
import java.util.Set;

import weka.classifiers.functions.LibLINEAR;
import weka.core.Attribute;
import weka.core.Instances;
import weka.core.SelectedTag;
import weka.core.SparseInstance;
import weka.core.converters.ArffSaver;
import cmu.arktweetnlp.Tagger;
import cmu.arktweetnlp.Tagger.TaggedToken;

import com.swabunga.spell.engine.SpellDictionary;
import com.swabunga.spell.engine.SpellDictionaryHashMap;
import com.swabunga.spell.engine.Word;
import com.swabunga.spell.event.SpellChecker;
import com.swabunga.spell.event.StringWordTokenizer;

import edu.stanford.nlp.ling.HasWord;
import edu.stanford.nlp.ling.TaggedWord;
import edu.stanford.nlp.process.DocumentPreprocessor;
import edu.stanford.nlp.tagger.maxent.MaxentTagger;


public class SentimentSystemTeamX extends SentimentSystem {

	public SentimentSystemTeamX(Set<Tweet> tweetList) {
		super(tweetList);
	}
	
public void train(String savename) throws IOException, ClassNotFoundException{
		
		System.out.println("Starting TeamX Train");
		System.out.println("Tweets: " +  tweetList.size());
		
		//load pos-tagger
		Tagger tagger = new Tagger();
    	tagger.loadModel("resources/tagger/model.20120919.txt");
    	MaxentTagger stanfordTagger = new MaxentTagger("resources/tagger/english-bidirectional-distsim.tagger");
  
    	//load sentiment lexica
    	Map<String, Double> afinnLexi = this.loadAFINN();
    	Map<String, Double> BingLiuLexi = this.loadBingLiu();
    	Map<String, Double> inquirerLexi = this.loadGeneralInquirer();
    	Map<String, Double> MPQALexi = this.loadMPQA();
    	Map<String, Double> senti140UniLexi = this.loadLexicon("sentiment140/unigrams-pmilexicon");
    	Map<String, Double> hashtagUniLexi = this.loadLexicon("hashtag/unigrams-pmilexicon");
    	Map<String, Double> senti140BiLexi = this.loadLexicon("sentiment140/bigrams-pmilexicon");
    	Map<String, Double> hashtagBiLexi = this.loadLexicon("hashtag/bigrams-pmilexicon");  
    	Map<String, Double> sentiWordNet = this.loadSentiWordNet();
    	
    	//load spell-checker
    	SpellDictionary dictionary = new SpellDictionaryHashMap(new File("resources/lexi/SpellChecker/english.0"), new File("resources/lexi/SpellChecker/phonet.en"));
    	SpellChecker spellChecker = new SpellChecker(dictionary);
    	
		int featurecount = 0;
		int tweetcount = 0;
		Map<String, Integer> nGramMap = new HashMap<String, Integer>();
		Map<String, Integer> CharNGramMap = new HashMap<String, Integer>();
		Map<String, Integer> clusterMap = new HashMap<String, Integer>();
		ArrayList<Attribute> attributeList = new ArrayList<Attribute>();
		
		//creating features
		for(Tweet tweet : tweetList){
		    
		    //preprocess and tag
			this.preProcessTweet(tweet);
			this.spellCorrection(spellChecker, tweet);
			this.tokenizeAndTag(tagger, stanfordTagger, tweet);
			this.negate(tweet);
			this.negateStanford(tweet);
			
			//get n-grams and set n-gram feature
			Set<String> nGramSet = this.getNGrams(tweet, 4);
			for (String nGram : nGramSet){
				if(!nGramMap.containsKey(nGram)){
					nGramMap.put(nGram, featurecount++);
					attributeList.add(new Attribute("NGRAM_" + nGram));
				}
			}
			
			//get char-n-grams and set char-n-gram feature
			Set<String> CharNGramSet = this.getCharNGrams(tweet);
			for (String nGram : CharNGramSet){
				if(!CharNGramMap.containsKey(nGram)){
					CharNGramMap.put(nGram, featurecount++);
					attributeList.add(new Attribute("CHARNGRAM_" + nGram));
				}
			}
			
			//get cluster and set cluster feature
			Set<String> clusterSet = this.getClusters(tweet);
			for(String cluster : clusterSet){
				if(!clusterMap.containsKey(cluster)){
					clusterMap.put(cluster, featurecount++);
					attributeList.add(new Attribute("CLUSTER_" + cluster));
				}
			}
		}
		
		//set lexica features
		//senti140Uni
		Attribute senti140UniTotalCountPos = new Attribute("senti140UniTotalCountPos");
		attributeList.add(senti140UniTotalCountPos);
		featurecount++;
		
		Attribute senti140UniTotalScorePos = new Attribute("senti140UniTotalScorePos");
		attributeList.add(senti140UniTotalScorePos);
		featurecount++;
		
		Attribute senti140UniMaxScorePos = new Attribute("senti140UniMaxScorePos");
		attributeList.add(senti140UniMaxScorePos);
		featurecount++;
		
		Attribute senti140UniLastScorePos = new Attribute("senti140UniLastScorePos");
		attributeList.add(senti140UniLastScorePos);
		featurecount++;
		
		Attribute senti140UniTotalCountNeg = new Attribute("senti140UniTotalCountNeg");
		attributeList.add(senti140UniTotalCountNeg);
		featurecount++;
		
		Attribute senti140UniTotalScoreNeg = new Attribute("senti140UniTotalScoreNeg");
		attributeList.add(senti140UniTotalScoreNeg);
		featurecount++;
		
		Attribute senti140UniMaxScoreNeg = new Attribute("senti140UniMaxScoreNeg");
		attributeList.add(senti140UniMaxScoreNeg);
		featurecount++;
		
		Attribute senti140UniLastScoreNeg = new Attribute("senti140UniLastScoreNeg");
		attributeList.add(senti140UniLastScoreNeg);
		featurecount++;
		
		//hashtagUni
		Attribute hashtagUniTotalCountPos = new Attribute("hashtagUniTotalCountPos");
		attributeList.add(hashtagUniTotalCountPos);
		featurecount++;
		
		Attribute hashtagUniTotalScorePos = new Attribute("hashtagUniTotalScorePos");
		attributeList.add(hashtagUniTotalScorePos);
		featurecount++;
		
		Attribute hashtagUniMaxScorePos = new Attribute("hashtagUniMaxScorePos");
		attributeList.add(hashtagUniMaxScorePos);
		featurecount++;
		
		Attribute hashtagUniLastScorePos = new Attribute("hashtagUniLastScorePos");
		attributeList.add(hashtagUniLastScorePos);
		featurecount++;
		
		Attribute hashtagUniTotalCountNeg = new Attribute("hashtagUniTotalCountNeg");
		attributeList.add(hashtagUniTotalCountNeg);
		featurecount++;
		
		Attribute hashtagUniTotalScoreNeg = new Attribute("hashtagUniTotalScoreNeg");
		attributeList.add(hashtagUniTotalScoreNeg);
		featurecount++;
		
		Attribute hashtagUniMaxScoreNeg = new Attribute("hashtagUniMaxScoreNeg");
		attributeList.add(hashtagUniMaxScoreNeg);
		featurecount++;
		
		Attribute hashtagUniLastScoreNeg = new Attribute("hashtagUniLastScoreNeg");
		attributeList.add(hashtagUniLastScoreNeg);
		featurecount++;
		
		//senti140Bi
		Attribute senti140BiTotalCountPos = new Attribute("senti140BiTotalCountPos");
		attributeList.add(senti140BiTotalCountPos);
		featurecount++;
		
		Attribute senti140BiTotalScorePos = new Attribute("senti140BiTotalScorePos");
		attributeList.add(senti140BiTotalScorePos);
		featurecount++;
		
		Attribute senti140BiMaxScorePos = new Attribute("senti140BiMaxScorePos");
		attributeList.add(senti140BiMaxScorePos);
		featurecount++;
		
		Attribute senti140BiLastScorePos = new Attribute("senti140BiLastScorePos");
		attributeList.add(senti140BiLastScorePos);
		featurecount++;
		
		Attribute senti140BiTotalCountNeg = new Attribute("senti140BiTotalCountNeg");
		attributeList.add(senti140BiTotalCountNeg);
		featurecount++;
		
		Attribute senti140BiTotalScoreNeg = new Attribute("senti140BiTotalScoreNeg");
		attributeList.add(senti140BiTotalScoreNeg);
		featurecount++;
		
		Attribute senti140BiMaxScoreNeg = new Attribute("senti140BiMaxScoreNeg");
		attributeList.add(senti140BiMaxScoreNeg);
		featurecount++;
		
		Attribute senti140BiLastScoreNeg = new Attribute("senti140BiLastScoreNeg");
		attributeList.add(senti140BiLastScoreNeg);
		featurecount++;
		
		//hashtagBi
		Attribute hashtagBiTotalCountPos = new Attribute("hashtagBiTotalCountPos");
		attributeList.add(hashtagBiTotalCountPos);
		featurecount++;
		
		Attribute hashtagBiTotalScorePos = new Attribute("hashtagBiTotalScorePos");
		attributeList.add(hashtagBiTotalScorePos);
		featurecount++;
		
		Attribute hashtagBiMaxScorePos = new Attribute("hashtagBiMaxScorePos");
		attributeList.add(hashtagBiMaxScorePos);
		featurecount++;
		
		Attribute hashtagBiLastScorePos = new Attribute("hashtagBiLastScorePos");
		attributeList.add(hashtagBiLastScorePos);
		featurecount++;
		
		Attribute hashtagBiTotalCountNeg = new Attribute("hashtagBiTotalCountNeg");
		attributeList.add(hashtagBiTotalCountNeg);
		featurecount++;
		
		Attribute hashtagBiTotalScoreNeg = new Attribute("hashtagBiTotalScoreNeg");
		attributeList.add(hashtagBiTotalScoreNeg);
		featurecount++;
		
		Attribute hashtagBiMaxScoreNeg = new Attribute("hashtagBiMaxScoreNeg");
		attributeList.add(hashtagBiMaxScoreNeg);
		featurecount++;
		
		Attribute hashtagBiLastScoreNeg = new Attribute("hashtagBiLastScoreNeg");
		attributeList.add(hashtagBiLastScoreNeg);
		featurecount++;
		
		//MPQA
		Attribute MPQATotalCountPos = new Attribute("MPQATotalCountPos");
		attributeList.add(MPQATotalCountPos);
		featurecount++;
		
		Attribute MPQATotalScorePos = new Attribute("MPQATotalScorePos");
		attributeList.add(MPQATotalScorePos);
		featurecount++;
		
		Attribute MPQAMaxScorePos = new Attribute("MPQAMaxScorePos");
		attributeList.add(MPQAMaxScorePos);
		featurecount++;
		
		Attribute MPQALastScorePos = new Attribute("MPQALastScorePos");
		attributeList.add(MPQALastScorePos);
		featurecount++;
		
		Attribute MPQATotalCountNeg = new Attribute("MPQATotalCountNeg");
		attributeList.add(MPQATotalCountNeg);
		featurecount++;
		
		Attribute MPQATotalScoreNeg = new Attribute("MPQATotalScoreNeg");
		attributeList.add(MPQATotalScoreNeg);
		featurecount++;
		
		Attribute MPQAMaxScoreNeg = new Attribute("MPQAMaxScoreNeg");
		attributeList.add(MPQAMaxScoreNeg);
		featurecount++;
		
		Attribute MPQALastScoreNeg = new Attribute("MPQALastScoreNeg");
		attributeList.add(MPQALastScoreNeg);
		featurecount++;
		
		//BingLiu
		Attribute BingLiuTotalCountPos = new Attribute("BingLiuTotalCountPos");
		attributeList.add(BingLiuTotalCountPos);
		featurecount++;
		
		Attribute BingLiuTotalScorePos = new Attribute("BingLiuTotalScorePos");
		attributeList.add(BingLiuTotalScorePos);
		featurecount++;
		
		Attribute BingLiuMaxScorePos = new Attribute("BingLiuMaxScorePos");
		attributeList.add(BingLiuMaxScorePos);
		featurecount++;
		
		Attribute BingLiuLastScorePos = new Attribute("BingLiuLastScorePos");
		attributeList.add(BingLiuLastScorePos);
		featurecount++;
		
		Attribute BingLiuTotalCountNeg = new Attribute("BingLiuTotalCountNeg");
		attributeList.add(BingLiuTotalCountNeg);
		featurecount++;
		
		Attribute BingLiuTotalScoreNeg = new Attribute("BingLiuTotalScoreNeg");
		attributeList.add(BingLiuTotalScoreNeg);
		featurecount++;
		
		Attribute BingLiuMaxScoreNeg = new Attribute("BingLiuMaxScoreNeg");
		attributeList.add(BingLiuMaxScoreNeg);
		featurecount++;
		
		Attribute BingLiuLastScoreNeg = new Attribute("BingLiuLastScoreNeg");
		attributeList.add(BingLiuLastScoreNeg);
		featurecount++;
		
		//AFINN
		Attribute afinnTotalCountPos = new Attribute("afinnTotalCountPos");
		attributeList.add(afinnTotalCountPos);
		featurecount++;
		Attribute afinnTotalScorePos = new Attribute("afinnTotalScorePos");
		attributeList.add(afinnTotalScorePos);
		featurecount++;
		Attribute afinnMaxScorePos = new Attribute("afinnMaxScorePos");
		attributeList.add(afinnMaxScorePos);
		featurecount++;
		Attribute afinnLastScorePos = new Attribute("afinnLastScorePos");
		attributeList.add(afinnLastScorePos);
		featurecount++;
		
		Attribute afinnTotalCountNeg = new Attribute("afinnTotalCountNeg");
		attributeList.add(afinnTotalCountNeg);
		featurecount++;
		Attribute afinnTotalScoreNeg= new Attribute("afinnTotalScoreNeg");
		attributeList.add(afinnTotalScoreNeg);
		featurecount++;
		Attribute afinnMaxScoreNeg = new Attribute("afinnMaxScoreNeg");
		attributeList.add(afinnMaxScoreNeg);
		featurecount++;
		Attribute afinnLastScoreNeg = new Attribute("afinnLastScoreNeg");
		attributeList.add(afinnLastScoreNeg);
		featurecount++;
		
		//WordNet
		Attribute wordNetTotalCountPos = new Attribute("wordNetTotalCountPos");
		attributeList.add(wordNetTotalCountPos);
		featurecount++;
		Attribute wordNetTotalScorePos = new Attribute("wordNetTotalScorePos");
		attributeList.add(wordNetTotalScorePos);
		featurecount++;
		Attribute wordNetMaxScorePos = new Attribute("wordNetMaxScorePos");
		attributeList.add(wordNetMaxScorePos);
		featurecount++;
		Attribute wordNetLastScorePos = new Attribute("wordNetLastScorePos");
		attributeList.add(wordNetLastScorePos);
		featurecount++;
		
		Attribute wordNetTotalCountNeg = new Attribute("wordNetTotalCountNeg");
		attributeList.add(wordNetTotalCountNeg);
		featurecount++;
		Attribute wordNetTotalScoreNeg = new Attribute("wordNetTotalScoreNeg");
		attributeList.add(wordNetTotalScoreNeg);
		featurecount++;
		Attribute wordNetMaxScoreNeg = new Attribute("wordNetMaxScoreNeg");
		attributeList.add(wordNetMaxScoreNeg);
		featurecount++;
		Attribute wordNetLastScoreNeg = new Attribute("wordNetLastScoreNeg");
		attributeList.add(wordNetLastScoreNeg);
		featurecount++;
		
		//Inquirer
		Attribute inquirerTotalCountPos = new Attribute("inquirerTotalCountPos");
		attributeList.add(inquirerTotalCountPos);
		featurecount++;
		Attribute inquirerTotalScorePos = new Attribute("inquirerTotalScorePos");
		attributeList.add(inquirerTotalScorePos);
		featurecount++;
		Attribute inquirerMaxScorePos = new Attribute("inquirerMaxScorePos");
		attributeList.add(inquirerMaxScorePos);
		featurecount++;
		Attribute inquirerLastScorePos = new Attribute("inquirerLastScorePos");
		attributeList.add(inquirerLastScorePos);
		featurecount++;
		
		Attribute inquirerTotalCountNeg = new Attribute("inquirerTotalCountNeg");
		attributeList.add(inquirerTotalCountNeg);
		featurecount++;
		Attribute inquirerTotalScoreNeg = new Attribute("inquirerTotalScoreNeg");
		attributeList.add(inquirerTotalScoreNeg);
		featurecount++;
		Attribute inquirerMaxScoreNeg = new Attribute("inquirerMaxScoreNeg");
		attributeList.add(inquirerMaxScoreNeg);
		featurecount++;
		Attribute inquirerLastScoreNeg = new Attribute("inquirerLastScoreNeg");
		attributeList.add(inquirerLastScoreNeg);
		featurecount++;
		
		//set class attribute
	    ArrayList<String> fvClassVal = new ArrayList<String>();
	    fvClassVal.add("positive");
	    fvClassVal.add("neutral");
	    fvClassVal.add("negative");
	    Attribute classAttribute = new Attribute("Class", fvClassVal);
	    attributeList.add(classAttribute);
		featurecount++;
		
		//creating instances with features
		Instances trainingSet = new Instances("test", attributeList, tweetList.size());
		trainingSet.setClassIndex(classAttribute.index());
		
		for(Tweet tweet : tweetList){
			SparseInstance instance = new SparseInstance(0);
			
			//n-gram feature
			Set<String> nGramSet = tweet.getnGramList();
			for (String nGram : nGramSet){
				Integer index = nGramMap.get(nGram);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//char-n-gram feature
			Set<String> CharNGramSet = tweet.getCharNGramList();
			for (String nGram : CharNGramSet){
				Integer index = CharNGramMap.get(nGram);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//cluster feature
			Set<String> clusterSet = tweet.getClusterList();
			for(String cluster : clusterSet){
				Integer index = clusterMap.get(cluster);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//lexica feature
			List<Double> senti140UniPos = this.getLexiScores(senti140UniLexi, tweet.getWordList(), false);
			instance.setValue(senti140UniTotalCountPos, senti140UniPos.get(0));
			instance.setValue(senti140UniTotalScorePos, senti140UniPos.get(1));
			instance.setValue(senti140UniMaxScorePos, senti140UniPos.get(2));
			instance.setValue(senti140UniLastScorePos, senti140UniPos.get(3));
			List<Double> hashtagUniPos = this.getLexiScores(hashtagUniLexi, tweet.getWordList(), false);
			instance.setValue(hashtagUniTotalCountPos, hashtagUniPos.get(0));
			instance.setValue(hashtagUniTotalScorePos, hashtagUniPos.get(1));
			instance.setValue(hashtagUniMaxScorePos, hashtagUniPos.get(2));
			instance.setValue(hashtagUniLastScorePos, hashtagUniPos.get(3));
			List<Double> senti140UniNeg = this.getLexiScores(senti140UniLexi, tweet.getWordList(), true);
			instance.setValue(senti140UniTotalCountNeg, senti140UniNeg.get(0));
			instance.setValue(senti140UniTotalScoreNeg, senti140UniNeg.get(1));
			instance.setValue(senti140UniMaxScoreNeg, senti140UniNeg.get(2));
			instance.setValue(senti140UniLastScoreNeg, senti140UniNeg.get(3));
			List<Double> hashtagUniNeg = this.getLexiScores(hashtagUniLexi, tweet.getWordList(), true);
			instance.setValue(hashtagUniTotalCountNeg, hashtagUniNeg.get(0));
			instance.setValue(hashtagUniTotalScoreNeg, hashtagUniNeg.get(1));
			instance.setValue(hashtagUniMaxScoreNeg, hashtagUniNeg.get(2));
			instance.setValue(hashtagUniLastScoreNeg, hashtagUniNeg.get(3));
			
			Set<String> biGramSet = this.getNGrams(tweet, 2, 2);
			List<Double> senti140BiPos = this.getLexiScoresBi(senti140BiLexi, biGramSet, false);
			instance.setValue(senti140BiTotalCountPos, senti140BiPos.get(0));
			instance.setValue(senti140BiTotalScorePos, senti140BiPos.get(1));
			instance.setValue(senti140BiMaxScorePos, senti140BiPos.get(2));
			instance.setValue(senti140BiLastScorePos, senti140BiPos.get(3));
			List<Double> hashtagBiPos = this.getLexiScoresBi(hashtagBiLexi, biGramSet, false);
			instance.setValue(hashtagBiTotalCountPos, hashtagBiPos.get(0));
			instance.setValue(hashtagBiTotalScorePos, hashtagBiPos.get(1));
			instance.setValue(hashtagBiMaxScorePos, hashtagBiPos.get(2));
			instance.setValue(hashtagBiLastScorePos, hashtagBiPos.get(3));
			List<Double> senti140BiNeg = this.getLexiScoresBi(senti140BiLexi, biGramSet, true);
			instance.setValue(senti140BiTotalCountNeg, senti140BiNeg.get(0));
			instance.setValue(senti140BiTotalScoreNeg, senti140BiNeg.get(1));
			instance.setValue(senti140BiMaxScoreNeg, senti140BiNeg.get(2));
			instance.setValue(senti140BiLastScoreNeg, senti140BiNeg.get(3));
			List<Double> hashtagBiNeg = this.getLexiScoresBi(hashtagBiLexi, biGramSet, true);
			instance.setValue(hashtagBiTotalCountNeg, hashtagBiNeg.get(0));
			instance.setValue(hashtagBiTotalScoreNeg, hashtagBiNeg.get(1));
			instance.setValue(hashtagBiMaxScoreNeg, hashtagBiNeg.get(2));
			instance.setValue(hashtagBiLastScoreNeg, hashtagBiNeg.get(3));
			
			List<Double> MPQAPos = this.getLexiScoresStandford(MPQALexi, tweet.getStanfordWordList(), false);
			instance.setValue(MPQATotalCountPos, MPQAPos.get(0));
			instance.setValue(MPQATotalScorePos, MPQAPos.get(1));
			instance.setValue(MPQAMaxScorePos, MPQAPos.get(2));
			instance.setValue(MPQALastScorePos, MPQAPos.get(3));
			List<Double> MPQANeg = this.getLexiScoresStandford(MPQALexi, tweet.getStanfordWordList(), true);
			instance.setValue(MPQATotalCountNeg, MPQANeg.get(0));
			instance.setValue(MPQATotalScoreNeg, MPQANeg.get(1));
			instance.setValue(MPQAMaxScoreNeg, MPQANeg.get(2));
			instance.setValue(MPQALastScoreNeg, MPQANeg.get(3));			
			
			List<Double> BingLiuPos = this.getLexiScores(BingLiuLexi, tweet.getWordList(), false);
			instance.setValue(BingLiuTotalCountPos, BingLiuPos.get(0));
			instance.setValue(BingLiuTotalScorePos, BingLiuPos.get(1));
			instance.setValue(BingLiuMaxScorePos, BingLiuPos.get(2));
			instance.setValue(BingLiuLastScorePos, BingLiuPos.get(3));
			List<Double> BingLiuNeg = this.getLexiScores(BingLiuLexi, tweet.getWordList(), true);
			instance.setValue(BingLiuTotalCountNeg, BingLiuNeg.get(0));
			instance.setValue(BingLiuTotalScoreNeg, BingLiuNeg.get(1));
			instance.setValue(BingLiuMaxScoreNeg, BingLiuNeg.get(2));
			instance.setValue(BingLiuLastScoreNeg, BingLiuNeg.get(3));
			
			List<Double> afinnScorePos = this.getLexiScores(afinnLexi, tweet.getWordList(), false);
			instance.setValue(afinnTotalCountPos, afinnScorePos.get(0));
			instance.setValue(afinnTotalScorePos, afinnScorePos.get(1));
			instance.setValue(afinnMaxScorePos, afinnScorePos.get(2));
			instance.setValue(afinnLastScorePos, afinnScorePos.get(3));
			List<Double> afinnScoreNeg = this.getLexiScores(afinnLexi, tweet.getWordList(), true);
			instance.setValue(afinnTotalCountNeg, afinnScoreNeg.get(0));
			instance.setValue(afinnTotalScoreNeg, afinnScoreNeg.get(1));
			instance.setValue(afinnMaxScoreNeg, afinnScoreNeg.get(2));
			instance.setValue(afinnLastScoreNeg, afinnScoreNeg.get(3));
			
			List<Double> wordNetScorePos = this.getLexiScoresStandford(sentiWordNet, tweet.getStanfordWordList(), false, true);
			instance.setValue(wordNetTotalCountPos, wordNetScorePos.get(0));
			instance.setValue(wordNetTotalScorePos, wordNetScorePos.get(1));
			instance.setValue(wordNetMaxScorePos, wordNetScorePos.get(2));
			instance.setValue(wordNetLastScorePos, wordNetScorePos.get(3));
			List<Double> wordNetScoreNeg = this.getLexiScoresStandford(sentiWordNet, tweet.getStanfordWordList(), true, true);
			instance.setValue(wordNetTotalCountNeg, wordNetScoreNeg.get(0));
			instance.setValue(wordNetTotalScoreNeg, wordNetScoreNeg.get(1));
			instance.setValue(wordNetMaxScoreNeg, wordNetScoreNeg.get(2));
			instance.setValue(wordNetLastScoreNeg, wordNetScoreNeg.get(3));
			
			List<Double> inquirerScorePos = this.getLexiScoresStandford(inquirerLexi, tweet.getStanfordWordList(), false);
			instance.setValue(inquirerTotalCountPos, inquirerScorePos.get(0));
			instance.setValue(inquirerTotalScorePos, inquirerScorePos.get(1));
			instance.setValue(inquirerMaxScorePos, inquirerScorePos.get(2));
			instance.setValue(inquirerLastScorePos, inquirerScorePos.get(3));
			List<Double> inquirerScoreNeg = this.getLexiScoresStandford(inquirerLexi, tweet.getStanfordWordList(), true);
			instance.setValue(inquirerTotalCountNeg, inquirerScoreNeg.get(0));
			instance.setValue(inquirerTotalScoreNeg, inquirerScoreNeg.get(1));
			instance.setValue(inquirerMaxScoreNeg, inquirerScoreNeg.get(2));
			instance.setValue(inquirerLastScoreNeg, inquirerScoreNeg.get(3));
			
			//set class attribute
			instance.setValue(classAttribute, tweet.getSentiment());
			
			trainingSet.add(instance);
		}
		
		//save features and training instances in .arff file
		ArffSaver saver = new ArffSaver();
		saver.setInstances(trainingSet);
		saver.setFile(new File("resources/arff/Trained-Features-" + "TeamX" + savename + ".arff"));
		saver.writeBatch();
		System.out.println("Trained-Features-" + "TeamX" + savename + " saved");
	}

	public Map<String,ClassificationResult> test(String nameOfTrain) throws Exception{
		System.out.println("Starting TeamX Test");
		System.out.println("Tweets: " +  this.tweetList.size());
		String trainname = "";
		if(!nameOfTrain.equals("")){
			trainname = nameOfTrain;
		}
		else{
			trainname = "Trained-Features-TeamX";
		}
		
		//load features and training instances from .arff file
		BufferedReader reader = new BufferedReader(new FileReader("resources/arff/" + trainname + ".arff"));
		Instances train = new Instances(reader);
		train.setClassIndex(train.numAttributes() - 1);
		reader.close();
		
		//load and setup classifier
		LibLINEAR classifier = new LibLINEAR();
	    classifier.setProbabilityEstimates(true);
	    classifier.setSVMType(new SelectedTag(0, LibLINEAR.TAGS_SVMTYPE));
	    classifier.setCost(0.05);
	    
	    //train classifier with instances
	    classifier.buildClassifier(train);
	    
	    //delete train instances, to use same features with test instances
		train.delete();
		
		//load pos-tagger
		Tagger tagger = new Tagger();
		tagger.loadModel("resources/tagger/model.20120919.txt");
		MaxentTagger stanfordTagger = new MaxentTagger("resources/tagger/english-bidirectional-distsim.tagger");
		
		//load sentiment lexica
		Map<String, Double> afinnLexi = this.loadAFINN();
		Map<String, Double> BingLiuLexi = this.loadBingLiu();
		Map<String, Double> inquirerLexi = this.loadGeneralInquirer();
		Map<String, Double> MPQALexi = this.loadMPQA();
		Map<String, Double> senti140UniLexi = this.loadLexicon("sentiment140/unigrams-pmilexicon");
		Map<String, Double> hashtagUniLexi = this.loadLexicon("hashtag/unigrams-pmilexicon");
		Map<String, Double> senti140BiLexi = this.loadLexicon("sentiment140/bigrams-pmilexicon");
		Map<String, Double> hashtagBiLexi = this.loadLexicon("hashtag/bigrams-pmilexicon");  
		Map<String, Double> sentiWordNet = this.loadSentiWordNet(); 
		
		
		//load spell-checker
		SpellDictionary dictionary = new SpellDictionaryHashMap(new File("resources/lexi/SpellChecker/english.0"), new File("resources/lexi/SpellChecker/phonet.en"));
		SpellChecker spellChecker = new SpellChecker(dictionary);
		
		Map<String, Integer> featureMap = new HashMap<String, Integer>();
		for (int i = 0; i < train.numAttributes(); i++){
			featureMap.put(train.attribute(i).name(), train.attribute(i).index());
		}
	    
		Map<String,ClassificationResult> resultMap = new HashMap<String, ClassificationResult>();
		for(Tweet tweet : tweetList){
		    
		    //preprocess and tag
			this.preProcessTweet(tweet);
			this.spellCorrection(spellChecker, tweet);
			this.tokenizeAndTag(tagger, stanfordTagger, tweet);
			this.negate(tweet);
			this.negateStanford(tweet);
			SparseInstance instance = new SparseInstance(0);
			
			//creating test instances with features
            //n-gram feature
			Set<String> nGramSet = this.getNGrams(tweet, 4);
			for (String nGram : nGramSet){
				Integer index = featureMap.get("NGRAM_" + nGram);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//char-n-gram feature
			Set<String> CharNGramSet = this.getCharNGrams(tweet);
			for (String nGram : CharNGramSet){
				Integer index = featureMap.get("CHARNGRAM_" + nGram);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//cluster feature			
			Set<String> clusterSet = this.getClusters(tweet);
			for(String cluster : clusterSet){
				Integer index = featureMap.get("CLUSTER_" + cluster);
				if(index != null){
					instance.setValue(index, 1);
				}
			}
			
			//lexica features
			List<Double> senti140UniPos = this.getLexiScores(senti140UniLexi, tweet.getWordList(), false);
			instance.setValue(featureMap.get("senti140UniTotalCountPos"), senti140UniPos.get(0));
			instance.setValue(featureMap.get("senti140UniTotalScorePos"), senti140UniPos.get(1));
			instance.setValue(featureMap.get("senti140UniMaxScorePos"), senti140UniPos.get(2));
			instance.setValue(featureMap.get("senti140UniLastScorePos"), senti140UniPos.get(3));
			List<Double> hashtagUniPos = this.getLexiScores(hashtagUniLexi, tweet.getWordList(), false);
			instance.setValue(featureMap.get("hashtagUniTotalCountPos"), hashtagUniPos.get(0));
			instance.setValue(featureMap.get("hashtagUniTotalScorePos"), hashtagUniPos.get(1));
			instance.setValue(featureMap.get("hashtagUniMaxScorePos"), hashtagUniPos.get(2));
			instance.setValue(featureMap.get("hashtagUniLastScorePos"), hashtagUniPos.get(3));
			List<Double> senti140UniNeg = this.getLexiScores(senti140UniLexi, tweet.getWordList(), true);
			instance.setValue(featureMap.get("senti140UniTotalCountNeg"), senti140UniNeg.get(0));
			instance.setValue(featureMap.get("senti140UniTotalScoreNeg"), senti140UniNeg.get(1));
			instance.setValue(featureMap.get("senti140UniMaxScoreNeg"), senti140UniNeg.get(2));
			instance.setValue(featureMap.get("senti140UniLastScoreNeg"), senti140UniNeg.get(3));
			List<Double> hashtagUniNeg = this.getLexiScores(hashtagUniLexi, tweet.getWordList(), true);
			instance.setValue(featureMap.get("hashtagUniTotalCountNeg"), hashtagUniNeg.get(0));
			instance.setValue(featureMap.get("hashtagUniTotalScoreNeg"), hashtagUniNeg.get(1));
			instance.setValue(featureMap.get("hashtagUniMaxScoreNeg"), hashtagUniNeg.get(2));
			instance.setValue(featureMap.get("hashtagUniLastScoreNeg"), hashtagUniNeg.get(3));
			
			Set<String> biGramSet = this.getNGrams(tweet, 2, 2);
			List<Double> senti140BiPos = this.getLexiScoresBi(senti140BiLexi, biGramSet, false);
			instance.setValue(featureMap.get("senti140BiTotalCountPos"), senti140BiPos.get(0));
			instance.setValue(featureMap.get("senti140BiTotalScorePos"), senti140BiPos.get(1));
			instance.setValue(featureMap.get("senti140BiMaxScorePos"), senti140BiPos.get(2));
			instance.setValue(featureMap.get("senti140BiLastScorePos"), senti140BiPos.get(3));
			List<Double> hashtagBiPos = this.getLexiScoresBi(hashtagBiLexi, biGramSet, false);
			instance.setValue(featureMap.get("hashtagBiTotalCountPos"), hashtagBiPos.get(0));
			instance.setValue(featureMap.get("hashtagBiTotalScorePos"), hashtagBiPos.get(1));
			instance.setValue(featureMap.get("hashtagBiMaxScorePos"), hashtagBiPos.get(2));
			instance.setValue(featureMap.get("hashtagBiLastScorePos"), hashtagBiPos.get(3));
			List<Double> senti140BiNeg = this.getLexiScoresBi(senti140BiLexi, biGramSet, true);
			instance.setValue(featureMap.get("senti140BiTotalCountNeg"), senti140BiNeg.get(0));
			instance.setValue(featureMap.get("senti140BiTotalScoreNeg"), senti140BiNeg.get(1));
			instance.setValue(featureMap.get("senti140BiMaxScoreNeg"), senti140BiNeg.get(2));
			instance.setValue(featureMap.get("senti140BiLastScoreNeg"), senti140BiNeg.get(3));
			List<Double> hashtagBiNeg = this.getLexiScoresBi(hashtagBiLexi, biGramSet, true);
			instance.setValue(featureMap.get("hashtagBiTotalCountNeg"), hashtagBiNeg.get(0));
			instance.setValue(featureMap.get("hashtagBiTotalScoreNeg"), hashtagBiNeg.get(1));
			instance.setValue(featureMap.get("hashtagBiMaxScoreNeg"), hashtagBiNeg.get(2));
			instance.setValue(featureMap.get("hashtagBiLastScoreNeg"), hashtagBiNeg.get(3));
			
			List<Double> MPQAPos = this.getLexiScoresStandford(MPQALexi, tweet.getStanfordWordList(), false);
			instance.setValue(featureMap.get("MPQATotalCountPos"), MPQAPos.get(0));
			instance.setValue(featureMap.get("MPQATotalScorePos"), MPQAPos.get(1));
			instance.setValue(featureMap.get("MPQAMaxScorePos"), MPQAPos.get(2));
			instance.setValue(featureMap.get("MPQALastScorePos"), MPQAPos.get(3));
			List<Double> MPQANeg = this.getLexiScoresStandford(MPQALexi, tweet.getStanfordWordList(), true);
			instance.setValue(featureMap.get("MPQATotalCountNeg"), MPQANeg.get(0));
			instance.setValue(featureMap.get("MPQATotalScoreNeg"), MPQANeg.get(1));
			instance.setValue(featureMap.get("MPQAMaxScoreNeg"), MPQANeg.get(2));
			instance.setValue(featureMap.get("MPQALastScoreNeg"), MPQANeg.get(3));
			
			List<Double> BingLiuPos = this.getLexiScores(BingLiuLexi, tweet.getWordList(), false);
			instance.setValue(featureMap.get("BingLiuTotalCountPos"), BingLiuPos.get(0));
			instance.setValue(featureMap.get("BingLiuTotalScorePos"), BingLiuPos.get(1));
			instance.setValue(featureMap.get("BingLiuMaxScorePos"), BingLiuPos.get(2));
			instance.setValue(featureMap.get("BingLiuLastScorePos"), BingLiuPos.get(3));
			List<Double> BingLiuNeg = this.getLexiScores(BingLiuLexi, tweet.getWordList(), true);
			instance.setValue(featureMap.get("BingLiuTotalCountNeg"), BingLiuNeg.get(0));
			instance.setValue(featureMap.get("BingLiuTotalScoreNeg"), BingLiuNeg.get(1));
			instance.setValue(featureMap.get("BingLiuMaxScoreNeg"), BingLiuNeg.get(2));
			instance.setValue(featureMap.get("BingLiuLastScoreNeg"), BingLiuNeg.get(3));
			
			List<Double> afinnScorePos = this.getLexiScores(afinnLexi, tweet.getWordList(), false);
			instance.setValue(featureMap.get("afinnTotalCountPos"), afinnScorePos.get(0));
			instance.setValue(featureMap.get("afinnTotalScorePos"), afinnScorePos.get(1));
			instance.setValue(featureMap.get("afinnMaxScorePos"), afinnScorePos.get(2));
			instance.setValue(featureMap.get("afinnLastScorePos"), afinnScorePos.get(3));
			List<Double> afinnScoreNeg = this.getLexiScores(afinnLexi, tweet.getWordList(), true);
			instance.setValue(featureMap.get("afinnTotalCountNeg"), afinnScoreNeg.get(0));
			instance.setValue(featureMap.get("afinnTotalScoreNeg"), afinnScoreNeg.get(1));
			instance.setValue(featureMap.get("afinnMaxScoreNeg"), afinnScoreNeg.get(2));
			instance.setValue(featureMap.get("afinnLastScoreNeg"), afinnScoreNeg.get(3));
			
			List<Double> wordNetScorePos = this.getLexiScoresStandford(sentiWordNet, tweet.getStanfordWordList(), false);
			instance.setValue(featureMap.get("wordNetTotalCountPos"), wordNetScorePos.get(0));
			instance.setValue(featureMap.get("wordNetTotalScorePos"), wordNetScorePos.get(1));
			instance.setValue(featureMap.get("wordNetMaxScorePos"), wordNetScorePos.get(2));
			instance.setValue(featureMap.get("wordNetLastScorePos"), wordNetScorePos.get(3));
			List<Double> wordNetScoreNeg = this.getLexiScoresStandford(sentiWordNet, tweet.getStanfordWordList(), true);
			instance.setValue(featureMap.get("wordNetTotalCountNeg"), wordNetScoreNeg.get(0));
			instance.setValue(featureMap.get("wordNetTotalScoreNeg"), wordNetScoreNeg.get(1));
			instance.setValue(featureMap.get("wordNetMaxScoreNeg"), wordNetScoreNeg.get(2));
			instance.setValue(featureMap.get("wordNetLastScoreNeg"), wordNetScoreNeg.get(3));
			
			List<Double> inquirerScorePos = this.getLexiScoresStandford(inquirerLexi, tweet.getStanfordWordList(), false);
			instance.setValue(featureMap.get("inquirerTotalCountPos"), inquirerScorePos.get(0));
			instance.setValue(featureMap.get("inquirerTotalScorePos"), inquirerScorePos.get(1));
			instance.setValue(featureMap.get("inquirerMaxScorePos"), inquirerScorePos.get(2));
			instance.setValue(featureMap.get("inquirerLastScorePos"), inquirerScorePos.get(3));
			List<Double> inquirerScoreNeg = this.getLexiScoresStandford(inquirerLexi, tweet.getStanfordWordList(), true);
			instance.setValue(featureMap.get("inquirerTotalCountNeg"), inquirerScoreNeg.get(0));
			instance.setValue(featureMap.get("inquirerTotalScoreNeg"), inquirerScoreNeg.get(1));
			instance.setValue(featureMap.get("inquirerMaxScoreNeg"), inquirerScoreNeg.get(2));
			instance.setValue(featureMap.get("inquirerLastScoreNeg"), inquirerScoreNeg.get(3));
			
			//add test instance to trained features
			train.add(instance);
			
            //classify Tweet
            double result = classifier.classifyInstance(train.lastInstance());
            double[] resultDistribution = classifier.distributionForInstance(train.lastInstance());
            resultMap.put(tweet.getTweetID(), new ClassificationResult(tweet, resultDistribution, result));
		}
		return resultMap;
	}
	
	//helper functions to preprocess and get features

	private void preProcessTweet(Tweet tweet){
		String rawTweet = tweet.getRawTweetString();
		rawTweet = rawTweet.toLowerCase();
		//filter Usernames
		rawTweet = rawTweet.replaceAll("@[^\\s]+", "");
		//filter Urls
		rawTweet = rawTweet.replaceAll("((www\\.[^\\s]+)|(https?://[^\\s]+))", "");
		tweet.setTweetString(rawTweet.trim());
	}

	private void tokenizeAndTag(Tagger tagger, MaxentTagger stanfordTagger, Tweet tweet) throws IOException{
		tweet.setWordList(tagger.tokenizeAndTag(tweet.getTweetString()));
		tweet.setStanfordWordList(tokenizeAndTagStanford(stanfordTagger, tweet));
	}
	
	private void spellCorrection(SpellChecker spellChecker, Tweet tweet){
		StringWordTokenizer tokenizer = new StringWordTokenizer(tweet.getTweetString());
		while(tokenizer.hasMoreWords()){
			List<Word> suggestions = spellChecker.getSuggestions(tokenizer.nextWord(), 0);
			if (!suggestions.isEmpty()){
				tokenizer.replaceWord(suggestions.get(0).getWord());
			}
		}
		tweet.setTweetString(tokenizer.getContext());
	}
	
	private Map<String, String> tokenizeAndTagStanford(MaxentTagger tagger, Tweet tweet) throws IOException {
		List<TaggedWord> wordListStanford2 = new ArrayList<TaggedWord>();
		Map<String, String> wordListStanford = new HashMap<String, String>();
		Reader reader = new StringReader(tweet.getTweetString());
		DocumentPreprocessor dp = new DocumentPreprocessor(reader);
		dp.setElementDelimiter("");
		Iterator<List<HasWord>> it = dp.iterator();
		while (it.hasNext()){
			wordListStanford2.addAll(tagger.tagSentence(it.next()));
		}
		reader.close();
		for (TaggedWord word : wordListStanford2) {
			wordListStanford.put(word.word(), word.tag());	
		}
		return wordListStanford;
	}
	
    private void negateStanford(Tweet tweet){
    	boolean neg = false;
    	for (Map.Entry<String, String> token : tweet.getStanfordWordList().entrySet()){
    		if(neg){
    			if(token.getKey().matches("^[.:;!?]$")){
    				neg = false;
    			}
    			else{
    				token.setValue(token.getKey() + "_NEG");
    			}
    		}
    		if(token.getKey().toLowerCase().matches("^(?:never|no|nothing|nowhere|noone|none|not|havent|hasnt|hadnt|cant|couldnt|shouldnt|wont|wouldnt|dont|doesnt|didnt|isnt|arent|aint)|.*n't")){
    			neg = true;
    		}
    	}
    }
    
    private List<Double> getLexiScoresStandford(Map<String,Double> lexi, Map<String, String> wordListStanford, boolean neg) {
        return getLexiScoresStandford(lexi, wordListStanford, neg, false);
    }
    
    private List<Double> getLexiScoresStandford(Map<String,Double> lexi, Map<String, String> wordListStanford, boolean neg, boolean wordnet) {
    	double totalCount = 0.0;
    	double totalScore = 0.0;
    	double maxScore = 0.0;
    	double lastScore = 0.0;
    	for (Map.Entry<String, String> token : wordListStanford.entrySet()){
    	    Double score;
    	    if (wordnet){
    	        if (neg){
    	            score = lexi.get(token.getKey() + "-");
    	        }
    	        else{
    	            score = lexi.get(token.getKey() + "+");
    	        }
    	    }
    	    else{
    	        score = lexi.get(token.getKey());
    	    }
    		if(score != null){
    			if (neg) score = score * -1;
    			if(score > 0) totalCount++;
    			totalScore = totalScore + score;
    			if(score > maxScore) maxScore = score;
    			if(score > 0) lastScore = score;
    		}
    	}
    	List<Double> scoreList = new ArrayList<Double>();
    	scoreList.add(totalCount);
    	scoreList.add(totalScore);
    	scoreList.add(maxScore);
    	scoreList.add(lastScore);
		return scoreList;
	}
    
    protected List<Double> getLexiScores(Map<String,Double> lexi, List<TaggedToken> wordList, boolean neg, boolean wordnet) {
    	double totalCount = 0.0;
    	double totalScore = 0.0;
    	double maxScore = 0.0;
    	double lastScore = 0.0;
    	for (TaggedToken token : wordList){
    		Double score;
    	    if (wordnet){
		        if (neg){
		            score = lexi.get(token.token + "-");
		        }
		        else{
		            score = lexi.get(token.token + "+");
		        }
    	    }
	        else{
	        	score = lexi.get(token.token);	        	
	        }
    		if(score != null){
    			if ((neg && score < 0) || (!neg && score > 0)){
    	             totalCount++;
    	             totalScore = totalScore + score;
    	             if (neg){
    	                 if(score < maxScore) maxScore = score;
                          
    	             }else{
    	                 if(score > maxScore) maxScore = score;	                 
    	             }
    	             lastScore = score; 
    			}
    		}
    	}
    	List<Double> scoreList = new ArrayList<Double>();
    	scoreList.add(totalCount);
    	scoreList.add(totalScore);
    	scoreList.add(maxScore);
    	scoreList.add(lastScore);
		return scoreList;
	}
    
    protected List<Double> getLexiScoresBi(Map<String,Double> lexi, Set<String> wordList, boolean neg) {
    	double totalCount = 0.0;
    	double totalScore = 0.0;
    	double maxScore = 0.0;
    	double lastScore = 0.0;
    	for (String token : wordList){
    		Double score;
	        score = lexi.get(token);	        	
    		if(score != null){
    			if ((neg && score < 0) || (!neg && score > 0)){
    	             totalCount++;
    	             totalScore = totalScore + score;
    	             if (neg){
    	                 if(score < maxScore) maxScore = score;
                          
    	             }else{
    	                 if(score > maxScore) maxScore = score;	                 
    	             }
    	             lastScore = score; 
    			}
    		}
    	}
    	List<Double> scoreList = new ArrayList<Double>();
    	scoreList.add(totalCount);
    	scoreList.add(totalScore);
    	scoreList.add(maxScore);
    	scoreList.add(lastScore);
		return scoreList;
	}

}
