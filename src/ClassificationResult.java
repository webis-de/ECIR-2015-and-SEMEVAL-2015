import java.util.ArrayList;

/**
 * Holds the result of the classification of one Tweet
 */
public class ClassificationResult {
    private Tweet tweet;
    private double[] resultDistribution;
    private double result;
    
    public ClassificationResult(Tweet t, double[] resultDis, double r){
        tweet = t;
        resultDistribution = resultDis;
        result = r;
    }
    
    public Tweet getTweet() {
        return tweet;
    }

    public double[] getResultDistribution() {
        return resultDistribution;
    }

    public double getResult() {
        return result;
    }
    
    public String getResultAsString() {
        ArrayList<String> classVal = new ArrayList<String>();
        classVal.add("positive");
        classVal.add("neutral");
        classVal.add("negative");
        return classVal.get((int)result);
    }
 }
