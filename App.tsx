
import React, { useState, useCallback, useMemo } from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, ScatterChart, Scatter, ZAxis } from 'recharts';
import type { DataRow, AnalysisResults, ModelReport, Prediction } from './types';
import * as geminiService from './services/geminiService';
import { HomeIcon, UploadIcon, ChartBarIcon, DocumentReportIcon, GlobeIcon } from './components/Icons';
import Loader from './components/Loader';
import Card from './components/Card';

type Page = 'home' | 'upload' | 'dashboard' | 'reports';

const App: React.FC = () => {
    const [page, setPage] = useState<Page>('home');
    const [isLoading, setIsLoading] = useState(false);
    const [loadingMessage, setLoadingMessage] = useState('');
    const [originalData, setOriginalData] = useState<DataRow[] | null>(null);
    const [processedData, setProcessedData] = useState<DataRow[] | null>(null);
    const [fileName, setFileName] = useState<string>('');
    const [analysisResults, setAnalysisResults] = useState<AnalysisResults | null>(null);
    const [modelReport, setModelReport] = useState<ModelReport | null>(null);
    const [predictions, setPredictions] = useState<Prediction[] | null>(null);

    // Model Config State
    const [targetColumn, setTargetColumn] = useState<string>('');
    const [featureColumns, setFeatureColumns] = useState<string[]>([]);
    const [modelType, setModelType] = useState<'Random Forest' | 'XGBoost'>('Random Forest');

    const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        const file = event.target.files?.[0];
        if (file) {
            setIsLoading(true);
            setLoadingMessage('Parsing CSV file...');
            setFileName(file.name);
            (window as any).Papa.parse(file, {
                header: true,
                skipEmptyLines: true,
                dynamicTyping: true,
                complete: (results: { data: DataRow[] }) => {
                    setOriginalData(results.data);
                    // Simple preprocessing: remove rows with null/undefined values
                    const cleanedData = results.data.filter(row => 
                        Object.values(row).every(val => val !== null && val !== undefined)
                    );
                    setProcessedData(cleanedData);
                    setPage('dashboard');
                    setIsLoading(false);
                },
                error: (error: Error) => {
                    console.error('CSV parsing error:', error);
                    alert('Error parsing CSV file. Please check the console.');
                    setIsLoading(false);
                }
            });
        }
    };
    
    const handleAnalyze = useCallback(async () => {
        if (!processedData) return;
        setIsLoading(true);
        setLoadingMessage('AI is analyzing your data...');
        const results = await geminiService.performDataAnalysis(processedData);
        setAnalysisResults(results);
        setIsLoading(false);
    }, [processedData]);


    const handleTrainModel = useCallback(async () => {
        if (!processedData || !targetColumn || featureColumns.length === 0) {
            alert("Please select a target variable and at least one feature.");
            return;
        }
        setIsLoading(true);
        setLoadingMessage(`AI is simulating a ${modelType} model...`);
        const result = await geminiService.generatePredictions(processedData, featureColumns, targetColumn, modelType);
        if (result) {
            setModelReport(result.report);
            setPredictions(result.predictions);
        } else {
            alert("Failed to generate predictions. Please check the console.");
        }
        setIsLoading(false);
    }, [processedData, targetColumn, featureColumns, modelType]);
    
    const handleDownloadPdfReport = useCallback(async () => {
        if (!analysisResults || !modelReport) {
            alert('Please analyze data and run a model first.');
            return;
        }
        setIsLoading(true);
        setLoadingMessage('Generating PDF report...');
        
        const narrative = await geminiService.generateReportNarrative(analysisResults, modelReport) || "Narrative generation failed.";
        
        const { jsPDF } = (window as any).jspdf;
        const doc = new jsPDF();

        doc.setFontSize(18);
        doc.text("Microplastic Pollution Risk Report", 14, 22);
        doc.setFontSize(11);
        doc.text(`Dataset: ${fileName}`, 14, 30);

        doc.setFontSize(12);
        doc.text("Analysis Narrative", 14, 45);
        doc.setFontSize(10);
        doc.splitTextToSize(narrative, 180).forEach((line: string, i: number) => {
            doc.text(line, 14, 52 + (i * 5));
        });

        const yPosAfterNarrative = 52 + (doc.splitTextToSize(narrative, 180).length * 5) + 10;
        
        (doc as any).autoTable({
            startY: yPosAfterNarrative,
            head: [['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score']],
            body: [[
                modelReport.modelType,
                modelReport.accuracy.toFixed(2),
                modelReport.precision.toFixed(2),
                modelReport.recall.toFixed(2),
                modelReport.f1.toFixed(2),
            ]],
            theme: 'striped',
        });
        
        const classReportData = Object.entries(modelReport.classificationReport)
            .map(([className, metrics]) => {
                const m = metrics as { precision: number; recall: number; 'f1-score': number; support: number; };
                return [
                    className,
                    m.precision.toFixed(2),
                    m.recall.toFixed(2),
                    (m['f1-score'] || 0).toFixed(2),
                    m.support
                ];
            });

        (doc as any).autoTable({
            head: [['Class', 'Precision', 'Recall', 'F1-Score', 'Support']],
            body: classReportData,
            theme: 'grid',
        });


        doc.save("microplastic_risk_report.pdf");
        setIsLoading(false);
    }, [analysisResults, modelReport, fileName]);

    const handleDownloadExcelReport = useCallback(() => {
        if (!predictions) {
            alert('Please generate predictions first.');
            return;
        }
        setIsLoading(true);
        setLoadingMessage('Generating Excel report...');
        const XLSX = (window as any).XLSX;
        const wb = XLSX.utils.book_new();
        const ws = XLSX.utils.json_to_sheet(predictions);
        XLSX.utils.book_append_sheet(wb, ws, "Predictions");
        if(originalData) {
            const ws_orig = XLSX.utils.json_to_sheet(originalData);
            XLSX.utils.book_append_sheet(wb, ws_orig, "Original Data");
        }
        XLSX.writeFile(wb, "microplastic_risk_predictions.xlsx");
        setIsLoading(false);
    }, [predictions, originalData]);

    const availableColumns = useMemo(() => {
        if (!processedData || processedData.length === 0) return [];
        return Object.keys(processedData[0]);
    }, [processedData]);
    
    const numericColumns = useMemo(() => {
        if (!processedData || processedData.length === 0) return [];
        return availableColumns.filter(col => typeof processedData[0][col] === 'number');
    }, [processedData, availableColumns]);

    const Sidebar = () => (
        <aside className="w-64 bg-gradient-to-b from-cyan-700 to-teal-600 text-white flex flex-col p-4 shadow-lg">
            <div className="flex items-center mb-8">
                <GlobeIcon className="w-10 h-10 mr-3 text-teal-300"/>
                <h1 className="text-xl font-bold leading-tight">Microplastic Risk System</h1>
            </div>
            <nav className="flex flex-col space-y-2">
                {[
                    { id: 'home', icon: HomeIcon, label: 'Home' },
                    { id: 'upload', icon: UploadIcon, label: 'Upload Data' },
                    { id: 'dashboard', icon: ChartBarIcon, label: 'Dashboard' },
                    { id: 'reports', icon: DocumentReportIcon, label: 'Reports' },
                ].map(item => (
                    <button
                        key={item.id}
                        onClick={() => setPage(item.id as Page)}
                        disabled={item.id !== 'home' && item.id !== 'upload' && !processedData}
                        className={`flex items-center px-4 py-3 rounded-md text-left transition-colors duration-200 ${
                            page === item.id ? 'bg-teal-500/50' : 'hover:bg-teal-500/30'
                        } ${item.id !== 'home' && item.id !== 'upload' && !processedData ? 'opacity-50 cursor-not-allowed' : ''}`}
                    >
                        <item.icon className="w-6 h-6 mr-4" />
                        <span className="text-lg">{item.label}</span>
                    </button>
                ))}
            </nav>
        </aside>
    );

    const MainContent = () => {
        if (isLoading) {
            return <div className="flex-grow flex items-center justify-center"><Loader message={loadingMessage} /></div>;
        }
        
        switch (page) {
            case 'home':
                return (
                    <div className="text-center">
                        <h2 className="text-4xl font-bold text-gray-800 mb-4">Welcome to the AI-Powered Microplastic Risk Assessment System</h2>
                        <p className="text-lg text-gray-600 max-w-3xl mx-auto mb-6">Leverage generative AI to analyze environmental data, predict pollution hotspots, and generate actionable insights. Upload your dataset to begin.</p>
                        <img src="https://picsum.photos/seed/environment/1000/400" alt="Environmental Sustainability" className="rounded-lg shadow-xl mx-auto mb-8"/>
                        <button onClick={() => setPage('upload')} className="bg-teal-600 text-white font-bold py-3 px-8 rounded-lg hover:bg-teal-700 transition-transform transform hover:scale-105 shadow-lg">Get Started</button>
                    </div>
                );
            case 'upload':
                 return (
                    <div className="w-full max-w-2xl mx-auto text-center">
                        <h2 className="text-3xl font-bold text-gray-800 mb-4">Upload Your Environmental Dataset</h2>
                        <p className="text-gray-600 mb-8">Please upload a CSV file. The system will automatically parse and clean the data for analysis.</p>
                        <div className="border-2 border-dashed border-gray-300 rounded-xl p-12 hover:border-teal-500 transition-colors">
                            <UploadIcon className="w-16 h-16 mx-auto text-gray-400 mb-4"/>
                            <input
                                type="file"
                                accept=".csv"
                                onChange={handleFileChange}
                                className="block w-full text-sm text-gray-500 file:mr-4 file:py-2 file:px-4 file:rounded-full file:border-0 file:text-sm file:font-semibold file:bg-teal-50 file:text-teal-700 hover:file:bg-teal-100"
                            />
                        </div>
                    </div>
                );
            case 'dashboard':
                 if (!processedData) return <div className="text-center text-xl text-gray-500">Please upload a dataset first.</div>;
                 return (
                    <div className="space-y-6">
                        <h2 className="text-3xl font-bold text-gray-800">Prediction Dashboard</h2>
                        
                        {/* Model Config Card */}
                        <Card title="Model Configuration">
                            <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Target Variable</label>
                                    <select value={targetColumn} onChange={e => setTargetColumn(e.target.value)} className="w-full p-2 border border-gray-300 rounded-md">
                                        <option value="">Select Target</option>
                                        {availableColumns.map(col => <option key={col} value={col}>{col}</option>)}
                                    </select>
                                </div>
                                <div className="md:col-span-2">
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Feature Variables</label>
                                    <select multiple value={featureColumns} onChange={e => setFeatureColumns(Array.from(e.target.selectedOptions, option => option.value))} className="w-full p-2 border border-gray-300 rounded-md h-32">
                                        {availableColumns.filter(c => c !== targetColumn).map(col => <option key={col} value={col}>{col}</option>)}
                                    </select>
                                </div>
                                <div>
                                    <label className="block text-sm font-medium text-gray-700 mb-1">Model Type</label>
                                    <div className="flex space-x-4 mt-2">
                                        <button onClick={() => setModelType('Random Forest')} className={`px-4 py-2 rounded-md ${modelType === 'Random Forest' ? 'bg-teal-600 text-white' : 'bg-gray-200'}`}>Random Forest</button>
                                        <button onClick={() => setModelType('XGBoost')} className={`px-4 py-2 rounded-md ${modelType === 'XGBoost' ? 'bg-teal-600 text-white' : 'bg-gray-200'}`}>XGBoost</button>
                                    </div>
                                </div>
                                <div className="md:col-span-3 flex justify-end items-center">
                                    <button onClick={handleAnalyze} disabled={!processedData || !!analysisResults} className="bg-gray-500 text-white font-bold py-2 px-6 rounded-lg hover:bg-gray-600 transition disabled:opacity-50 mr-4">
                                        {analysisResults ? 'Analysis Complete' : 'Run Data Analysis'}
                                    </button>
                                    <button onClick={handleTrainModel} className="bg-teal-600 text-white font-bold py-2 px-6 rounded-lg hover:bg-teal-700 transition">Generate Predictions</button>
                                </div>
                            </div>
                        </Card>

                        {/* Analysis & Predictions */}
                        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                            {analysisResults && (
                                <Card title="Data Analysis Insights">
                                    <ul className="list-disc list-inside space-y-2 text-gray-700">
                                        {analysisResults.insights.map((insight, i) => <li key={i}>{insight}</li>)}
                                    </ul>
                                </Card>
                            )}

                            {modelReport && (
                                <Card title="Model Performance">
                                    <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-center">
                                        <div className="p-2 bg-teal-50 rounded-lg">
                                            <p className="text-sm text-gray-500">Model</p>
                                            <p className="text-xl font-bold text-teal-700">{modelReport.modelType}</p>
                                        </div>
                                        <div className="p-2 bg-blue-50 rounded-lg">
                                            <p className="text-sm text-gray-500">Accuracy</p>
                                            <p className="text-xl font-bold text-blue-700">{modelReport.accuracy.toFixed(2)}</p>
                                        </div>
                                        <div className="p-2 bg-green-50 rounded-lg">
                                            <p className="text-sm text-gray-500">Precision</p>
                                            <p className="text-xl font-bold text-green-700">{modelReport.precision.toFixed(2)}</p>
                                        </div>
                                        <div className="p-2 bg-yellow-50 rounded-lg">
                                            <p className="text-sm text-gray-500">F1-Score</p>
                                            <p className="text-xl font-bold text-yellow-700">{modelReport.f1.toFixed(2)}</p>
                                        </div>
                                    </div>
                                </Card>
                            )}
                        </div>

                        {predictions && (
                            <Card title="Sample Predictions">
                                <div className="overflow-x-auto">
                                    <table className="min-w-full divide-y divide-gray-200">
                                        <thead className="bg-gray-50">
                                            <tr>
                                                {Object.keys(predictions[0]).map(key => (
                                                    <th key={key} className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">{key.replace(/_/g, ' ')}</th>
                                                ))}
                                            </tr>
                                        </thead>
                                        <tbody className="bg-white divide-y divide-gray-200">
                                            {predictions.slice(0, 10).map((row, i) => (
                                                <tr key={i}>
                                                    {Object.entries(row).map(([key, value], j) => (
                                                        <td key={j} className="px-6 py-4 whitespace-nowrap text-sm text-gray-800">
                                                            {key === 'Predicted_Risk' ? (
                                                                <span className={`px-2 inline-flex text-xs leading-5 font-semibold rounded-full ${
                                                                    (value as any) === 'High' ? 'bg-red-100 text-red-800' :
                                                                    (value as any) === 'Medium' ? 'bg-yellow-100 text-yellow-800' :
                                                                    'bg-green-100 text-green-800'
                                                                }`}>
                                                                    {value as any}
                                                                </span>
                                                            ) : String(value as any)}
                                                        </td>
                                                    ))}
                                                </tr>
                                            ))}
                                        </tbody>
                                    </table>
                                </div>
                            </Card>
                        )}
                        
                        {analysisResults && numericColumns.includes('latitude') && numericColumns.includes('longitude') && (
                            <Card title="Geographic Risk Distribution (Simulated)">
                                <ResponsiveContainer width="100%" height={400}>
                                    <ScatterChart margin={{ top: 20, right: 20, bottom: 20, left: 20 }}>
                                        <CartesianGrid />
                                        <XAxis type="number" dataKey="longitude" name="longitude" />
                                        <YAxis type="number" dataKey="latitude" name="latitude" />
                                        <ZAxis type="category" dataKey="Predicted_Risk" name="Risk" />
                                        <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                                        <Legend />
                                        <Scatter name="Low Risk" data={(predictions || []).filter(p => p.Predicted_Risk === 'Low')} fill="#22c55e" />
                                        <Scatter name="Medium Risk" data={(predictions || []).filter(p => p.Predicted_Risk === 'Medium')} fill="#f59e0b" />
                                        <Scatter name="High Risk" data={(predictions || []).filter(p => p.Predicted_Risk === 'High')} fill="#ef4444" />
                                    </ScatterChart>
                                </ResponsiveContainer>
                            </Card>
                        )}
                    </div>
                 );
            case 'reports':
                return (
                     <div className="space-y-6">
                        <h2 className="text-3xl font-bold text-gray-800">Generate & Download Reports</h2>
                        <p className="text-gray-600">Download comprehensive reports based on your analysis and predictions.</p>
                         <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                            <Card title="PDF Report">
                                <p className="text-gray-600 mb-4">A summary report including analysis narrative and model performance metrics.</p>
                                <button onClick={handleDownloadPdfReport} disabled={!analysisResults || !modelReport} className="w-full bg-red-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-red-700 transition disabled:opacity-50 disabled:cursor-not-allowed">Download PDF</button>
                            </Card>
                            <Card title="Excel Report">
                                <p className="text-gray-600 mb-4">Raw prediction data and original dataset for further analysis.</p>
                                <button onClick={handleDownloadExcelReport} disabled={!predictions} className="w-full bg-green-600 text-white font-bold py-3 px-6 rounded-lg hover:bg-green-700 transition disabled:opacity-50 disabled:cursor-not-allowed">Download Excel</button>
                            </Card>
                        </div>
                    </div>
                );
        }
    };


    return (
        <div className="flex h-screen bg-gray-50 font-sans">
            <Sidebar />
            <main className="flex-1 p-8 overflow-y-auto">
                <div className="max-w-7xl mx-auto">
                    <MainContent />
                </div>
            </main>
        </div>
    );
};

export default App;
