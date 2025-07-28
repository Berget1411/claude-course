"use client";

import { useState, useRef, useEffect } from "react";

interface Document {
  doc_id: string;
  filename: string;
  chunk_count: number;
  upload_time: string;
}

export default function Page() {
  const [message, setMessage] = useState("");
  const [response, setResponse] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [documents, setDocuments] = useState<Document[]>([]);
  const [isUploading, setIsUploading] = useState(false);
  const [useTools, setUseTools] = useState(true);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Load documents on component mount
  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    try {
      const res = await fetch("http://127.0.0.1:8000/documents");
      if (res.ok) {
        const docs = await res.json();
        setDocuments(docs);
      }
    } catch (error) {
      console.error("Error loading documents:", error);
    }
  };

  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      alert("Please select a PDF file");
      return;
    }

    setIsUploading(true);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const res = await fetch("http://127.0.0.1:8000/upload", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const result = await res.json();
        alert(
          `Successfully uploaded ${file.name}! Created ${result.chunk_count} chunks.`
        );
        loadDocuments(); // Reload document list
      } else {
        const error = await res.json();
        alert(`Upload failed: ${error.detail}`);
      }
    } catch (error) {
      console.error("Upload error:", error);
      alert("Upload failed due to network error");
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) {
        fileInputRef.current.value = "";
      }
    }
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!message.trim()) return;

    setIsLoading(true);
    setResponse("");

    try {
      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        body: JSON.stringify({
          message,
          use_tools: useTools,
        }),
        headers: {
          "Content-Type": "application/json",
        },
      });

      if (!res.body) {
        throw new Error("No response body");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });

        // Parse Server-Sent Events
        const lines = chunk.split("\n");
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const dataStr = line.slice(6); // Remove 'data: ' prefix
            if (dataStr.trim()) {
              try {
                const data = JSON.parse(dataStr);
                if (data.type === "text" && data.content) {
                  setResponse((prev) => prev + data.content);
                } else if (data.type === "tool_start" && data.tool_name) {
                  setResponse(
                    (prev) => prev + `\n🔧 Using tool: ${data.tool_name}\n`
                  );
                } else if (data.type === "tool_complete") {
                  setResponse((prev) => prev + "✅ Tool execution completed\n");
                } else if (data.type === "done") {
                  setIsLoading(false);
                } else if (data.type === "error") {
                  setResponse((prev) => prev + `\n❌ Error: ${data.content}\n`);
                }
              } catch (e) {
                console.error("Error parsing JSON:", e);
              }
            }
          }
        }
      }
    } catch (error) {
      console.error("Error:", error);
      setResponse("Error occurred while streaming response");
    } finally {
      setIsLoading(false);
    }
  };

  const clearConversation = async () => {
    try {
      await fetch("http://127.0.0.1:8000/clear", { method: "POST" });
      setResponse("");
      setMessage("");
    } catch (error) {
      console.error("Error clearing conversation:", error);
    }
  };

  return (
    <div style={{ padding: "20px", maxWidth: "1200px", margin: "0 auto" }}>
      <h1>📚 PDF RAG Chatbot</h1>
      <p>
        Upload PDF documents and chat with them using AI! Ask questions about
        your documents or get help with coding libraries.
      </p>

      {/* Document Upload Section */}
      <div
        style={{
          marginBottom: "20px",
          padding: "15px",
          border: "2px dashed #ddd",
          borderRadius: "8px",
          backgroundColor: "#f8f9fa",
        }}
      >
        <h3>📄 Upload Documents</h3>
        <input
          ref={fileInputRef}
          type='file'
          accept='.pdf'
          onChange={handleFileUpload}
          disabled={isUploading}
          style={{ marginBottom: "10px" }}
        />
        {isUploading && <div>Uploading and processing...</div>}

        {documents.length > 0 && (
          <div style={{ marginTop: "10px" }}>
            <h4>Loaded Documents:</h4>
            <ul style={{ margin: "5px 0", paddingLeft: "20px" }}>
              {documents.map((doc, idx) => (
                <li key={idx} style={{ marginBottom: "5px" }}>
                  <strong>{doc.filename}</strong> - {doc.chunk_count} chunks
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      {/* Chat Section */}
      <form onSubmit={handleSubmit} style={{ marginBottom: "20px" }}>
        <div style={{ marginBottom: "10px" }}>
          <label
            style={{
              display: "flex",
              alignItems: "center",
              marginBottom: "10px",
            }}
          >
            <input
              type='checkbox'
              checked={useTools}
              onChange={(e) => setUseTools(e.target.checked)}
              style={{ marginRight: "8px" }}
            />
            Enable tools (document search, library lookup)
          </label>

          <textarea
            value={message}
            onChange={(e) => setMessage(e.target.value)}
            placeholder={`Try asking:
- "What is this document about?"
- "Search for information about machine learning"
- "How do I use FastAPI for streaming?"
- "Show me anthropic SDK examples"`}
            rows={4}
            style={{
              width: "100%",
              padding: "10px",
              fontSize: "16px",
              border: "1px solid #ccc",
              borderRadius: "4px",
              resize: "vertical",
            }}
            disabled={isLoading}
          />
        </div>
        <div style={{ display: "flex", gap: "10px" }}>
          <button
            type='submit'
            disabled={isLoading || !message.trim()}
            style={{
              padding: "10px 20px",
              fontSize: "16px",
              backgroundColor: isLoading ? "#ccc" : "#007bff",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: isLoading ? "not-allowed" : "pointer",
            }}
          >
            {isLoading ? "🤔 Thinking..." : "💬 Send"}
          </button>

          <button
            type='button'
            onClick={clearConversation}
            disabled={isLoading}
            style={{
              padding: "10px 20px",
              fontSize: "16px",
              backgroundColor: "#6c757d",
              color: "white",
              border: "none",
              borderRadius: "4px",
              cursor: "pointer",
            }}
          >
            🗑️ Clear
          </button>
        </div>
      </form>

      {/* Response Section */}
      <div
        style={{
          marginTop: "20px",
          padding: "15px",
          border: "1px solid #ddd",
          borderRadius: "8px",
          minHeight: "200px",
          backgroundColor: "#f9f9f9",
          whiteSpace: "pre-wrap",
          fontFamily: "Monaco, 'Courier New', monospace",
          fontSize: "14px",
          lineHeight: "1.5",
        }}
      >
        {response ||
          (isLoading
            ? "🤔 AI is thinking..."
            : documents.length > 0
            ? "Ready to chat! Ask me anything about your uploaded documents or coding libraries."
            : "Upload a PDF document to get started, or ask about coding libraries!")}
      </div>

      {/* Quick Examples */}
      {documents.length > 0 && !isLoading && (
        <div style={{ marginTop: "15px", fontSize: "14px", color: "#666" }}>
          <strong>💡 Quick examples:</strong>
          <div
            style={{
              display: "flex",
              flexWrap: "wrap",
              gap: "5px",
              marginTop: "5px",
            }}
          >
            {[
              "What are the main topics in this document?",
              "Search for methodology",
              "How do I use FastAPI?",
              "Show me Anthropic examples",
            ].map((example, idx) => (
              <button
                key={idx}
                onClick={() => setMessage(example)}
                disabled={isLoading}
                style={{
                  padding: "5px 10px",
                  fontSize: "12px",
                  backgroundColor: "#e9ecef",
                  border: "1px solid #ddd",
                  borderRadius: "15px",
                  cursor: "pointer",
                }}
              >
                {example}
              </button>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
