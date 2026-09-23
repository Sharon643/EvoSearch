import { useState } from "react";
import axios from "axios";
import "./App.css";

const API_URL = "http://127.0.0.1:8000";

function App() {
  const [query, setQuery] = useState("");
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const handleResearch = async () => {
    if (!query.trim()) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const response = await axios.post(`${API_URL}/research`, {
        query: query.trim(),
      });

      setResult(response.data);
    } catch (err) {
      console.error(err);
      setError(
        err.response?.data?.detail ||
          "Unable to connect to the EvoSearch backend."
      );
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      handleResearch();
    }
  };

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-mark">E</span>
          <span>EvoSearch</span>
        </div>

        <span className="status">
          <span className="status-dot"></span>
          AI Research Agent
        </span>
      </header>

      <main className="main">
        <section className="hero">
          <p className="eyebrow">SELF-IMPROVING RESEARCH</p>

          <h1>
            Research smarter.
            <br />
            <span>Find better evidence.</span>
          </h1>

          <p className="subtitle">
            EvoSearch breaks complex questions into research tasks,
            searches multiple sources, evaluates evidence, and
            synthesizes an answer.
          </p>

          <div className="search-box">
            <textarea
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="What do you want to research?"
              rows={3}
              disabled={loading}
            />

            <button
              onClick={handleResearch}
              disabled={loading || !query.trim()}
            >
              {loading ? "Researching..." : "Research"}
            </button>
          </div>

          <p className="hint">
            Press Enter to research
          </p>
        </section>

        {loading && (
          <section className="loading-card">
            <div className="spinner"></div>

            <div>
              <h3>Researching your question</h3>
              <p>
                EvoSearch is searching, evaluating, and synthesizing
                evidence...
              </p>
            </div>
          </section>
        )}

        {error && (
          <section className="error-card">
            <strong>Research failed</strong>
            <p>{error}</p>
            <small>
              Make sure the FastAPI backend is running on port 8000.
            </small>
          </section>
        )}

        {result && !loading && (
          <section className="results">
            <div className="result-header">
              <div>
                <p className="eyebrow">RESEARCH RESULT</p>
                <h2>{result.query}</h2>
              </div>

              <div className="validation">
                <span
                  className={
                    result.validation === "VALID"
                      ? "validation-dot valid"
                      : "validation-dot invalid"
                  }
                ></span>

                {result.validation}
              </div>
            </div>

            <div className="answer-card">
              <h3>Answer</h3>

              <div className="answer">
                {result.answer.split("\n").map((line, index) => (
                  <p key={index}>
                    {line || "\u00A0"}
                  </p>
                ))}
              </div>
            </div>

            <div className="research-meta">
              <div>
                <span>Queries</span>
                <strong>{result.queries_used?.length || 0}</strong>
              </div>

              <div>
                <span>Sources</span>
                <strong>{result.sources?.length || 0}</strong>
              </div>

              <div>
                <span>Retries</span>
                <strong>{result.retry_count || 0}</strong>
              </div>
            </div>

            {result.sources?.length > 0 && (
              <div className="sources-section">
                <div className="section-heading">
                  <p className="eyebrow">EVIDENCE</p>
                  <h2>Sources</h2>
                </div>

                <div className="sources">
                  {result.sources.map((source, index) => (
                    <article className="source-card" key={source.url}>
                      <div className="source-number">
                        {index + 1}
                      </div>

                      <div className="source-content">
                        <h3>{source.title}</h3>

                        <p>{source.content}</p>

                        <div className="source-footer">
                          <a
                            href={source.url}
                            target="_blank"
                            rel="noopener noreferrer"
                          >
                            Open source →
                          </a>

                          <span>
                            Score {source.score}
                          </span>
                        </div>
                      </div>
                    </article>
                  ))}
                </div>
              </div>
            )}

            <div className="queries-section">
              <div className="section-heading">
                <p className="eyebrow">RESEARCH PROCESS</p>
                <h2>Queries used</h2>
              </div>

              <div className="query-list">
                {result.queries_used?.map((item, index) => (
                  <div className="query-item" key={index}>
                    <span>{index + 1}</span>
                    {item}
                  </div>
                ))}
              </div>
            </div>
          </section>
        )}
      </main>
    </div>
  );
}

export default App;