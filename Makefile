run_cli:
	python -m src.app.cli analyze --cv sample/sample_cv.txt --role "Senior AI Engineer" --region Global --lang en

run_ui:
	streamlit run src/app/streamlit_app.py
