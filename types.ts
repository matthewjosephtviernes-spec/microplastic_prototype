
export type DataRow = Record<string, string | number>;

export interface DescriptiveStat {
  mean: number;
  std: number;
  min: number;
  '25%': number;
  '50%': number;
  // FIX: Corrected typo from Mnumber to number
  '75%': number;
  max: number;
}

export interface AnalysisResults {
  descriptiveStats: Record<string, DescriptiveStat>;
  correlationMatrix: Record<string, Record<string, number>>;
  insights: string[];
  numericColumns: string[];
}

export interface ClassificationReport {
  [key: string]: {
    precision: number;
    recall: number;
    'f1-score': number;
    support: number;
  };
}

export interface ModelReport {
  modelType: string;
  accuracy: number;
  precision: number;
  recall: number;
  f1: number;
  classificationReport: ClassificationReport;
}

export interface Prediction extends DataRow {
  Predicted_Risk: 'Low' | 'Medium' | 'High';
}