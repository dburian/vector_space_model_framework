package cz.dburian;

import org.terrier.matching.models.TF_IDF;
import org.terrier.matching.models.WeightingModel;
import org.terrier.matching.models.WeightingModelLibrary;

class TfIdfLengthPivoted extends WeightingModel {
  private static final String name = "TfIdfPivoted";
  protected double slope;
  // private TF_IDF tfidf_model;

  public TfIdfLengthPivoted() {
    super();
    // this.tfidf_model = new TF_IDF();
  }

  @Override
  public String getInfo() {
    return name;
  }

  public TfIdfLengthPivoted(double slope) {
    this();
    this.slope = slope;
  }

  @Override
  public double score(double term_freq, double docLength) {
    double tf = 1 + WeightingModelLibrary.log(term_freq);
    double idf = WeightingModelLibrary.log(numberOfDocuments/documentFrequency+1);
    double tfidf = tf * idf;
    return WeightingModelLibrary.tf_pivoted(tfidf, slope, docLength, averageDocumentLength);
  }

}
