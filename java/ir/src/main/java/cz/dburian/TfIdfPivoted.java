package cz.dburian;

import org.terrier.matching.models.TF_IDF;
import org.terrier.matching.models.WeightingModel;
import org.terrier.matching.models.WeightingModelLibrary;

class TfIdfPivoted extends WeightingModel {
  private static final String name = "TfIdfPivoted";
  protected double slope;
  private TF_IDF tfidf_model;

  public TfIdfPivoted() {
    super();
    this.tfidf_model = new TF_IDF();
  }

  @Override
  public String getInfo() {
    return name;
  }

  public TfIdfPivoted(double slope) {
    this();
    this.slope = slope;
  }

  @Override
  public double score(double tf, double docLength) {
    double tfidf = tfidf_model.score(tf, docLength);
    return WeightingModelLibrary.tf_pivoted(tfidf, slope, docLength, averageDocumentLength);
  }

}
