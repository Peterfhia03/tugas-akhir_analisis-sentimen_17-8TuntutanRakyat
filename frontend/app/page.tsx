"use client";

import { useState } from "react";
import axios from "axios";

interface PredictionResponse {
  raw_text: string;
  clean_text: string;
  model_used: string;
  prediction: string;
  confidence: number;
  probabilities: {
    Negatif: number;
    Netral: number;
    Positif: number;
  };
}

interface BatchPredictionRow {
  "Teks Preprocessing": string;
  Sentimen: string;
  Confidence: number;
}

const ROWS_PER_PAGE = 10;

export default function Home() {
  const [text, setText] = useState("");
  const [model, setModel] = useState("IndoBERT-BiLSTM");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [file, setFile] = useState<File | null>(null);
  const [batchResult, setBatchResult] = useState<BatchPredictionRow[]>([]);
  const [currentPage, setCurrentPage] = useState(1);
  const [loading, setLoading] = useState(false);

  const totalPages = Math.ceil(batchResult.length / ROWS_PER_PAGE);

  const currentRows = batchResult.slice(
    (currentPage - 1) * ROWS_PER_PAGE,
    currentPage * ROWS_PER_PAGE,
  );

  const handlePredict = async () => {
    setLoading(true);
    setResult(null);
    setBatchResult([]);
    setCurrentPage(1);

    try {
      if (file) {
        const formData = new FormData();
        formData.append("file", file);
        formData.append("model", model);

        const res = await axios.post<BatchPredictionRow[]>(
          "http://localhost:8000/predict-batch",
          formData,
          { headers: { "Content-Type": "multipart/form-data" } },
        );

        setBatchResult(res.data);
      } else {
        const res = await axios.post<PredictionResponse>(
          "http://localhost:8000/predict",
          { text, model },
        );

        setResult(res.data);
      }
    } catch (error) {
      alert("Terjadi kesalahan saat memproses data.");
    }

    setLoading(false);
  };

  const handleDownload = () => {
    if (batchResult.length === 0) return;

    const headers: (keyof BatchPredictionRow)[] = [
      "Teks Preprocessing",
      "Sentimen",
      "Confidence",
    ];

    const rows = batchResult.map((row) => headers.map((header) => row[header]));

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((r) => r.join(","))].join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute("download", "hasil_prediksi_batch.csv");
    document.body.appendChild(link);
    link.click();
  };

  return (
    <div className="min-h-screen bg-[#065F46] text-white">
      <section className="px-6 py-16 max-w-6xl mx-auto">
        <h1 className="text-3xl md:text-5xl font-bold text-pink-400 text-center">
          Analisis Sentimen 17+8 Tuntutan Rakyat
        </h1>

        <p className="mt-6 text-center max-w-3xl mx-auto text-gray-200 text-sm md:text-base">
          Platform ini membantu menganalisis sentimen publik terhadap 17+8
          Tuntutan Rakyat menggunakan model hibrida IndoBERT‑BiLSTM berbasis
          data autentik dari Media Sosial X. Sistem mendukung prediksi teks
          tunggal maupun batch melalui file CSV untuk eksplorasi opini publik
          secara komputasional.
        </p>

        <div className="mt-12 bg-white/90 backdrop-blur-sm text-black rounded-2xl shadow-2xl p-6 md:p-10">
          <textarea
            className="w-full border p-4 rounded-xl focus:ring-2 focus:ring-pink-400"
            rows={4}
            placeholder="Masukkan teks untuk dianalisis..."
            value={text}
            onChange={(e) => setText(e.target.value)}
            disabled={file !== null}
          />

          <div className="mt-4">
            <p className="text-xs text-gray-600 mb-2">
              Atau upload file CSV (tepat 1 kolom berisi teks)
            </p>
            <input
              type="file"
              accept=".csv"
              onChange={(e) => {
                if (e.target.files) {
                  setFile(e.target.files[0]);
                  setText("");
                }
              }}
            />
          </div>

          {/* <select
            className="w-full mt-4 p-3 border rounded-xl"
            value={model}
            onChange={(e) => setModel(e.target.value)}
          >
            <option>IndoBERT</option>
            <option>BiLSTM</option>
            <option>IndoBERT-BiLSTM</option>
          </select> */}

          <button
            onClick={handlePredict}
            disabled={loading}
            className="mt-6 w-full bg-pink-500 hover:bg-pink-600 text-white py-3 rounded-xl font-bold"
          >
            {loading ? "Memproses..." : "Prediksi"}
          </button>
        </div>
      </section>

      {(result || batchResult.length > 0) && (
        <section className="bg-[#064E3B] px-6 py-16">
          <div className="max-w-[1300px] mx-auto">
            {/* SINGLE RESULT */}
            {result && (
              <div className="bg-white/90 text-black rounded-2xl shadow-xl p-8">
                <h2 className="text-2xl font-bold text-[#065F46] mb-6">
                  Hasil Analisis
                </h2>

                <div className="grid md:grid-cols-2 gap-6">
                  <div>
                    <p className="text-sm text-gray-500">Teks Preprocessing</p>
                    <p>{result.clean_text}</p>

                    <p className="mt-4 text-sm text-gray-500">Model</p>
                    <p>{result.model_used}</p>
                  </div>

                  <div className="bg-gray-50 p-6 rounded-xl text-center">
                    <p className="text-sm text-gray-500">Sentimen</p>
                    <p className="text-3xl font-bold text-pink-500">
                      {result.prediction}
                    </p>

                    <div className="mt-4 w-full bg-gray-200 rounded-full h-3">
                      <div
                        className="bg-pink-500 h-3 rounded-full"
                        style={{ width: `${result.confidence * 100}%` }}
                      />
                    </div>
                    <p className="mt-2 font-semibold">
                      {(result.confidence * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
                <div className="mt-6 grid md:grid-cols-3 gap-4 text-center">
                  <div className="bg-red-100 p-4 rounded-xl">
                    <p className="text-black">Negatif</p>
                    <p className="font-bold text-black">
                      {(result.probabilities.Negatif * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="bg-gray-200 p-4 rounded-xl">
                    <p className="text-black">Netral</p>
                    <p className="font-bold text-black">
                      {(result.probabilities.Netral * 100).toFixed(2)}%
                    </p>
                  </div>
                  <div className="bg-green-200 p-4 rounded-xl">
                    <p className="text-black">Positif</p>
                    <p className="font-bold text-black">
                      {(result.probabilities.Positif * 100).toFixed(2)}%
                    </p>
                  </div>
                </div>
              </div>
            )}

            {/* BATCH RESULT */}
            {batchResult.length > 0 && (
              <div className="mt-10 bg-white/90 text-black rounded-2xl shadow-xl p-10 overflow-x-auto">
                <h2 className="text-3xl font-bold mb-5 text-[#065F46] text-center">
                  Hasil Prediksi
                </h2>

                <table className="min-w-full border">
                  <thead className="bg-pink-500 text-white">
                    <tr>
                      <th className="p-3 border">Teks Preprocessing</th>
                      <th className="p-3 border">Sentimen</th>
                      <th className="p-3 border">Confidence</th>
                    </tr>
                  </thead>
                  <tbody>
                    {currentRows.map((row, index) => (
                      <tr key={index}>
                        <td className="p-3 border">
                          {row["Teks Preprocessing"]}
                        </td>
                        <td className="p-3 border font-semibold">
                          {row["Sentimen"]}
                        </td>
                        <td className="p-3 border">
                          {(row["Confidence"] * 100).toFixed(2)}%
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>

                {/* PAGINATION */}
                <div className="flex justify-end items-center mt-6 space-x-2">
                  <button
                    disabled={currentPage === 1}
                    onClick={() => setCurrentPage(currentPage - 1)}
                    className="px-3 py-1 bg-gray-200 rounded"
                  >
                    ←
                  </button>

                  <span className="font-semibold">
                    {currentPage} / {totalPages}
                  </span>

                  <button
                    disabled={currentPage === totalPages}
                    onClick={() => setCurrentPage(currentPage + 1)}
                    className="px-3 py-1 bg-gray-200 rounded"
                  >
                    →
                  </button>
                </div>
                <button
                  onClick={handleDownload}
                  className="bg-pink-500 hover:bg-pink-600 text-white px-7 py-2 rounded-xl font-semibold transition"
                >
                  Unduh CSV
                </button>
              </div>
            )}
          </div>
        </section>
      )}
    </div>
  );
}
