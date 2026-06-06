import asyncio
import json
import logging
import sys
import threading
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from platformdirs import user_config_dir

from gurrt.core.models import download_models
from gurrt.core.pipeline import VideoRag

# Suppress warnings
logging.disable(logging.WARNING)

# Configure stdout encoding
if sys.stdout:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, RuntimeError):
        pass

# Configure logging for the app
logger = logging.getLogger(__name__)


class GurrtGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("🌿 Gurrt - Video Understanding Tool")
        self.root.geometry("900x750")
        self.root.resizable(True, True)

        # Configure style
        self.root.configure(bg="#f0f0f0")
        style = ttk.Style()
        style.theme_use("clam")

        # Initialize variables
        self.config_dir = Path(user_config_dir("gurrt"))
        self.config_dir.mkdir(exist_ok=True, parents=True)
        self.config_file = self.config_dir / "config.json"
        self.rag = None
        self.is_processing = False

        # Initialize logs_text placeholder before setup_ui
        self.logs_text = None

        # Build UI
        self.setup_ui()
        self.check_configuration()

    def setup_ui(self):
        """Setup the main UI components"""

        # Main container with tabs
        notebook = ttk.Notebook(self.root)
        notebook.pack(fill="both", expand=True, padx=10, pady=10)

        # Tab 1: Setup & Configuration
        setup_frame = ttk.Frame(notebook)
        notebook.add(setup_frame, text="⚙️ Setup")
        self.setup_setup_tab(setup_frame)

        # Tab 2: Indexing
        index_frame = ttk.Frame(notebook)
        notebook.add(index_frame, text="📹 Index Video")
        self.setup_index_tab(index_frame)

        # Tab 3: Query
        query_frame = ttk.Frame(notebook)
        notebook.add(query_frame, text="❓ Ask Question")
        self.setup_query_tab(query_frame)

        # Tab 4: Logs/Output
        logs_frame = ttk.Frame(notebook)
        notebook.add(logs_frame, text="📊 Status & Logs")
        self.setup_logs_tab(logs_frame)

    def setup_setup_tab(self, parent):
        """Setup configuration tab"""
        main_frame = ttk.Frame(parent, padding="20")
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_frame, text="Configuration & Setup", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))

        # API Keys Section
        api_frame = ttk.LabelFrame(
            main_frame, text="API Keys Configuration", padding="15"
        )
        api_frame.pack(fill="x", pady=10)

        # Groq API Key
        ttk.Label(api_frame, text="Groq API Key:", font=("Arial", 10)).pack(
            anchor="w", pady=(0, 5)
        )
        self.groq_var = tk.StringVar()
        groq_entry = ttk.Entry(
            api_frame, textvariable=self.groq_var, show="*", width=50
        )
        groq_entry.pack(fill="x", pady=(0, 15))
        groq_info = ttk.Label(
            api_frame,
            text="Get your key at: https://console.groq.com/docs/models",
            font=("Arial", 8),
            foreground="blue",
        )
        groq_info.pack(anchor="w", pady=(0, 15))

        # Supermemory API Key
        ttk.Label(api_frame, text="Supermemory API Key:", font=("Arial", 10)).pack(
            anchor="w", pady=(0, 5)
        )
        self.supermemory_var = tk.StringVar()
        supermemory_entry = ttk.Entry(
            api_frame, textvariable=self.supermemory_var, show="*", width=50
        )
        supermemory_entry.pack(fill="x", pady=(0, 15))
        supermemory_info = ttk.Label(
            api_frame,
            text="Get your key at: https://supermemory.ai/docs/integrations/supermemory-sdk",
            font=("Arial", 8),
            foreground="blue",
        )
        supermemory_info.pack(anchor="w")

        # Load existing config if available
        self.load_api_keys()

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=20)

        save_config_btn = ttk.Button(
            button_frame, text="💾 Save Configuration", command=self.save_configuration
        )
        save_config_btn.pack(side="left", padx=5)

        # Models Section
        models_frame = ttk.LabelFrame(main_frame, text="Model Management", padding="15")
        models_frame.pack(fill="x", pady=10)

        models_info = ttk.Label(
            models_frame,
            text="Download and cache required AI models locally for offline use.",
            font=("Arial", 9),
            foreground="gray",
        )
        models_info.pack(anchor="w", pady=(0, 15))

        download_btn = ttk.Button(
            models_frame, text="⬇️ Download Models", command=self.download_models_async
        )
        download_btn.pack(fill="x", pady=5)

        self.models_progress = ttk.Progressbar(models_frame, mode="indeterminate")
        self.models_progress.pack(fill="x", pady=5)

        self.models_status = ttk.Label(models_frame, text="Ready", font=("Arial", 9))
        self.models_status.pack(anchor="w")

    def setup_index_tab(self, parent):
        """Setup video indexing tab"""
        main_frame = ttk.Frame(parent, padding="20")
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_frame, text="Index Video for Search", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))

        # Video Selection
        video_frame = ttk.LabelFrame(main_frame, text="Select Video File", padding="15")
        video_frame.pack(fill="x", pady=10)

        ttk.Label(video_frame, text="Video Path:", font=("Arial", 10)).pack(
            anchor="w", pady=(0, 5)
        )

        path_frame = ttk.Frame(video_frame)
        path_frame.pack(fill="x", pady=(0, 15))

        self.video_path_var = tk.StringVar()
        video_entry = ttk.Entry(path_frame, textvariable=self.video_path_var, width=50)
        video_entry.pack(side="left", fill="x", expand=True, padx=(0, 10))

        browse_btn = ttk.Button(path_frame, text="📂 Browse", command=self.browse_video)
        browse_btn.pack(side="left")

        # Indexing Options
        options_frame = ttk.LabelFrame(
            main_frame, text="Indexing Options", padding="15"
        )
        options_frame.pack(fill="x", pady=10)

        ttk.Label(options_frame, text="Processing Method:", font=("Arial", 10)).pack(
            anchor="w", pady=(0, 10)
        )

        self.method_var = tk.StringVar(value="groq")
        ttk.Radiobutton(
            options_frame,
            text="Groq (Cloud-based, Faster)",
            variable=self.method_var,
            value="groq",
        ).pack(anchor="w", pady=5)

        ttk.Radiobutton(
            options_frame,
            text="Ollama (Local, Requires Ollama Setup)",
            variable=self.method_var,
            value="ollama",
        ).pack(anchor="w", pady=5)

        # Ollama model selection
        ollama_frame = ttk.Frame(options_frame)
        ollama_frame.pack(fill="x", pady=(10, 0))

        ttk.Label(
            ollama_frame, text="Ollama Model (if selected):", font=("Arial", 9)
        ).pack(anchor="w", pady=(0, 5))
        self.ollama_model_var = tk.StringVar(value="gemma2")
        ollama_entry = ttk.Entry(
            ollama_frame, textvariable=self.ollama_model_var, width=30
        )
        ollama_entry.pack(anchor="w")

        # Index Button
        index_btn = ttk.Button(
            main_frame, text="🚀 Start Indexing", command=self.index_video_async
        )
        index_btn.pack(fill="x", pady=20)

        # Progress
        self.index_progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.index_progress.pack(fill="x", pady=5)

        self.index_status = ttk.Label(main_frame, text="Ready", font=("Arial", 9))
        self.index_status.pack(anchor="w")

        # Info
        info_frame = ttk.LabelFrame(main_frame, text="ℹ️ Information", padding="10")
        info_frame.pack(fill="both", expand=True, pady=10)

        info_text = """How Indexing Works:

1. The system extracts key frames from your video
2. Captions are generated for each frame using BLIP
3. Audio is extracted and transcribed using Faster-Whisper
4. All content is embedded into vector space for semantic search
5. Results are stored for quick querying

This process may take several minutes depending on video length."""

        info_label = ttk.Label(
            info_frame, text=info_text, font=("Arial", 9), justify="left"
        )
        info_label.pack(anchor="nw", fill="both", expand=True)

    def setup_query_tab(self, parent):
        """Setup query tab"""
        main_frame = ttk.Frame(parent, padding="20")
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_frame,
            text="Ask Questions About Your Video",
            font=("Arial", 16, "bold"),
        )
        title_label.pack(pady=(0, 20))

        # Query Input
        query_frame = ttk.LabelFrame(main_frame, text="Your Question", padding="15")
        query_frame.pack(fill="both", expand=True, pady=10)

        ttk.Label(
            query_frame, text="What would you like to know?", font=("Arial", 10)
        ).pack(anchor="nw", pady=(0, 10))

        self.query_text = tk.Text(
            query_frame, height=5, wrap="word", font=("Arial", 10)
        )
        self.query_text.pack(fill="both", expand=True, pady=(0, 10))

        # Buttons
        button_frame = ttk.Frame(main_frame)
        button_frame.pack(fill="x", pady=10)

        ask_btn = ttk.Button(button_frame, text="🔍 Ask", command=self.ask_query_async)
        ask_btn.pack(side="left", padx=5)

        clear_btn = ttk.Button(
            button_frame,
            text="🗑️ Clear",
            command=lambda: self.query_text.delete("1.0", tk.END),
        )
        clear_btn.pack(side="left", padx=5)

        # Response
        response_frame = ttk.LabelFrame(main_frame, text="Answer", padding="15")
        response_frame.pack(fill="both", expand=True, pady=10)

        self.response_text = scrolledtext.ScrolledText(
            response_frame, height=10, wrap="word", font=("Arial", 10)
        )
        self.response_text.pack(fill="both", expand=True)

        # Progress
        self.query_progress = ttk.Progressbar(main_frame, mode="indeterminate")
        self.query_progress.pack(fill="x", pady=5)

        self.query_status = ttk.Label(main_frame, text="Ready", font=("Arial", 9))
        self.query_status.pack(anchor="w")

    def setup_logs_tab(self, parent):
        """Setup logs/status tab"""
        main_frame = ttk.Frame(parent, padding="20")
        main_frame.pack(fill="both", expand=True)

        # Title
        title_label = ttk.Label(
            main_frame, text="Status & Logs", font=("Arial", 16, "bold")
        )
        title_label.pack(pady=(0, 20))

        # Logs Display
        self.logs_text = scrolledtext.ScrolledText(
            main_frame, height=20, wrap="word", font=("Courier", 9), bg="#f5f5f5"
        )
        self.logs_text.pack(fill="both", expand=True)

        # Clear button
        clear_btn = ttk.Button(
            main_frame,
            text="🗑️ Clear Logs",
            command=lambda: self.logs_text.delete("1.0", tk.END),
        )
        clear_btn.pack(pady=10)

    def log_message(self, message):
        """Log a message to the logs tab"""
        if self.logs_text is None:
            return  # logs_text not yet initialized, skip logging
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.logs_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.logs_text.see(tk.END)
        self.root.update()

    def check_configuration(self):
        """Check if configuration exists"""
        if not self.config_file.exists():
            self.log_message(
                "⚠️ Configuration not found. Please set up API keys in the Setup tab."
            )

    def load_api_keys(self):
        """Load existing API keys from config"""
        if self.config_file.exists():
            try:
                with open(self.config_file, "r") as f:
                    config = json.load(f)
                    self.groq_var.set(config.get("GROQ_API_KEY", ""))
                    self.supermemory_var.set(config.get("SUPERMEMORY_API_KEY", ""))
                    self.log_message("✅ Configuration loaded successfully.")
            except Exception as e:
                self.log_message(f"❌ Error loading configuration: {str(e)}")

    def save_configuration(self):
        """Save API configuration"""
        groq_key = self.groq_var.get().strip()
        supermemory_key = self.supermemory_var.get().strip()

        if not groq_key or not supermemory_key:
            messagebox.showerror("Error", "Please enter both API keys.")
            return

        try:
            with open(self.config_file, "w") as f:
                json.dump(
                    {
                        "GROQ_API_KEY": groq_key,
                        "SUPERMEMORY_API_KEY": supermemory_key,
                    },
                    f,
                    indent=2,
                )

            self.log_message("✅ Configuration saved successfully!")
            messagebox.showinfo("Success", "Configuration saved successfully!")
        except Exception as e:
            self.log_message(f"❌ Error saving configuration: {str(e)}")
            messagebox.showerror("Error", f"Failed to save configuration: {str(e)}")

    def download_models_async(self):
        """Download models in a separate thread"""
        if self.is_processing:
            messagebox.showwarning("Warning", "A process is already running.")
            return

        thread = threading.Thread(target=self.download_models)
        thread.daemon = True
        thread.start()

    def download_models(self):
        """Download required models"""
        self.is_processing = True
        self.models_progress.start()
        self.models_status.config(text="Downloading models...")
        self.log_message("⏳ Starting model download...")

        try:
            cache_dir = self.config_dir / "models"
            cache_dir.mkdir(exist_ok=True, parents=True)
            download_models(cache_dir)

            self.log_message("✅ Models downloaded successfully!")
            self.models_status.config(text="✅ Models downloaded successfully!")
            messagebox.showinfo("Success", "Models downloaded and cached successfully!")
        except Exception as e:
            self.log_message(f"❌ Error downloading models: {str(e)}")
            messagebox.showerror("Error", f"Failed to download models: {str(e)}")
            self.models_status.config(text="❌ Error during download")
        finally:
            self.models_progress.stop()
            self.is_processing = False

    def browse_video(self):
        """Browse and select a video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video Files", "*.mp4 *.avi *.mov *.mkv"),
                ("All Files", "*.*"),
            ],
        )
        if filename:
            self.video_path_var.set(filename)
            self.log_message(f"📹 Video selected: {filename}")

    def index_video_async(self):
        """Index video in a separate thread"""
        if self.is_processing:
            messagebox.showwarning("Warning", "A process is already running.")
            return

        video_path = self.video_path_var.get()
        method = self.method_var.get()
        thread = threading.Thread(target=self.index_video, args=(video_path, method))
        thread.daemon = True
        thread.start()

    def index_video(self, video_path, method):
        """Index the selected video"""
        self.is_processing = True
        self.index_progress.start()
        self.index_status.config(text="Indexing in progress...")

        try:
            self.log_message(f"🚀 Starting video indexing: {video_path}")
            self.log_message(f"📊 Method: {method}")

            self.rag = VideoRag(reset=True)

            if method == "groq":
                self.log_message("📹 Extracting frames and generating captions...")
                self.rag.index_video(video_path=video_path)
                self.log_message("🎵 Extracting and transcribing audio...")
                self.rag.index_audio(video_path=video_path)
            else:  # ollama
                ollama_model = self.ollama_model_var.get().strip() or "gemma2"
                self.log_message(f"🤖 Using Ollama model: {ollama_model}")
                self.log_message("📹 Extracting frames and generating captions...")
                self.rag.index_video_ollama(
                    video_path=video_path, model_name=ollama_model
                )
                self.log_message("🎵 Extracting and transcribing audio...")
                self.rag.index_audio(video_path=video_path)

            self.log_message("✅ Video indexed successfully!")
            self.index_status.config(
                text="✅ Video indexed successfully! You can now ask questions."
            )
            messagebox.showinfo(
                "Success",
                "Video indexed successfully! You can now ask questions in the 'Ask Question' tab.",
            )
        except Exception as e:
            self.log_message(f"❌ Error during indexing: {str(e)}")
            messagebox.showerror("Error", f"Failed to index video: {str(e)}")
            self.index_status.config(text="❌ Error during indexing")
        finally:
            self.index_progress.stop()
            self.is_processing = False

    def ask_query_async(self):
        """Ask a query in a separate thread"""
        if self.is_processing:
            messagebox.showwarning("Warning", "A process is already running.")
            return

        query = self.query_text.get("1.0", tk.END).strip()
        if not query:
            messagebox.showerror("Error", "Please enter a question.")
            return

        if self.rag is None:
            messagebox.showerror(
                "Error", "Please index a video first in the 'Index Video' tab."
            )
            return

        thread = threading.Thread(target=self.ask_query)
        thread.daemon = True
        thread.start()

    def ask_query(self):
        """Ask a query about the indexed video"""
        self.is_processing = True
        self.query_progress.start()
        self.query_status.config(text="Thinking...")
        self.response_text.delete("1.0", tk.END)

        query = self.query_text.get("1.0", tk.END).strip()

        try:
            self.log_message(f"❓ Query: {query}")
            self.log_message("⏳ Processing query...")

            if self.rag is not None:
                response = asyncio.run(self.rag.ask(query=query))
            else:
                raise RuntimeError("VideoRag not initialized")

            self.response_text.insert(tk.END, response)
            self.log_message("✅ Response generated successfully!")
            self.query_status.config(text="✅ Response ready")
        except Exception as e:
            error_msg = f"Error processing query: {str(e)}"
            self.response_text.insert(tk.END, f"❌ {error_msg}")
            self.log_message(f"❌ {error_msg}")
            self.query_status.config(text="❌ Error processing query")
        finally:
            self.query_progress.stop()
            self.is_processing = False


def main():
    root = tk.Tk()
    GurrtGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
