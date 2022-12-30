package cz.dburian;

import org.terrier.matching.models.TF_IDF;
import org.terrier.matching.models.WeightingModel;
import org.terrier.matching.models.WeightingModelLibrary;

class TfIdfRobertsonPivoted extends WeightingModel {
  private static final String name = "TfIdfRobertsonPivoted";
  private TF_IDF tfidf_model;
  private static final double k1 = 1.2d;
  protected double slope;

  public TfIdfRobertsonPivoted() {
    super();

    this.tfidf_model = new TF_IDF();
  }

  public TfIdfRobertsonPivoted(double slope) {
    this();
    this.slope = slope;
  }

  @Override
  public String getInfo() {
    return name;
  }

  @Override
  public double score(double tf, double docLength) {
    double tfidf = tfidf_model.score(tf, docLength);
    return WeightingModelLibrary.tf_robertson(tfidf, slope, docLength, averageDocumentLength, k1);
  }

}
