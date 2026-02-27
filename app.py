import os
import uuid
import logging
import sys
import io
import json

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, send_from_directory
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
from sqlalchemy import text

from models import db, Question, Test, TestQuestion, Scan, Response
from utils import (
    generate_test_pdf,
    process_scan_image,
    import_questions_from_xlsx,
    import_questions_from_json,
    export_questions_to_xlsx,
    build_omr_template,
)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_SCAN_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "pdf"}
ALLOWED_XLSX_EXTENSIONS = {"xlsx", "xls"}
ALLOWED_JSON_EXTENSIONS = {"json"}
VALID_ANSWER_TYPES = {"yes_no", "likert", "custom"}


def create_app():
    app = Flask(__name__)
    env_secret = os.environ.get("SECRET_KEY")
    is_production = os.environ.get("FLASK_ENV") == "production"
    is_test_runtime = "pytest" in sys.modules
    if is_production and not env_secret and not is_test_runtime:
        raise RuntimeError("SECRET_KEY precisa estar definida fora de ambiente de desenvolvimento/teste.")
    app.config["SECRET_KEY"] = env_secret or "dev-secret-key"
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL", "sqlite:///quiz.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    app.logger.setLevel(logging.INFO)
    csrf = CSRFProtect(app)

    db.init_app(app)

    with app.app_context():
        db.create_all()

    # ── Questions ───────────────────────────────────────────────────────────

    @app.route("/")
    def index():
        questions = Question.query.order_by(Question.created_at.desc()).all()
        tests = Test.query.order_by(Test.created_at.desc()).all()
        return render_template("index.html", questions=questions, tests=tests)

    @app.route("/questions")
    def questions_list():
        questions = Question.query.order_by(Question.created_at.desc()).all()
        return render_template("questions.html", questions=questions)

    @app.route("/questions/new", methods=["GET", "POST"])
    def question_new():
        if request.method == "POST":
            text = request.form["text"].strip()
            answer_type = request.form["answer_type"].strip()
            correct_answer = request.form.get("correct_answer", "").strip() or None
            custom_options = request.form.get("custom_options", "").strip() or None
            errors = _validate_question_payload(
                text=text,
                answer_type=answer_type,
                correct_answer=correct_answer,
                custom_options=custom_options,
            )
            if errors:
                for err in errors:
                    flash(err, "danger")
                return render_template("question_form.html", question=None), 400
            q = Question(
                text=text,
                answer_type=answer_type,
                correct_answer=correct_answer,
                custom_options=custom_options,
            )
            db.session.add(q)
            db.session.commit()
            flash("Pergunta criada com sucesso!", "success")
            return redirect(url_for("questions_list"))
        return render_template("question_form.html", question=None)

    @app.route("/questions/<int:qid>/edit", methods=["GET", "POST"])
    def question_edit(qid):
        q = Question.query.get_or_404(qid)
        if request.method == "POST":
            text = request.form["text"].strip()
            answer_type = request.form["answer_type"].strip()
            correct_answer = request.form.get("correct_answer", "").strip() or None
            custom_options = request.form.get("custom_options", "").strip() or None
            errors = _validate_question_payload(
                text=text,
                answer_type=answer_type,
                correct_answer=correct_answer,
                custom_options=custom_options,
            )
            if errors:
                for err in errors:
                    flash(err, "danger")
                q.text = text
                q.answer_type = answer_type
                q.correct_answer = correct_answer
                q.custom_options = custom_options
                return render_template("question_form.html", question=q), 400
            q.text = text
            q.answer_type = answer_type
            q.correct_answer = correct_answer
            q.custom_options = custom_options
            db.session.commit()
            flash("Pergunta atualizada!", "success")
            return redirect(url_for("questions_list"))
        return render_template("question_form.html", question=q)

    @app.route("/questions/<int:qid>/delete", methods=["POST"])
    def question_delete(qid):
        q = Question.query.get_or_404(qid)
        db.session.delete(q)
        db.session.commit()
        flash("Pergunta excluída.", "info")
        return redirect(url_for("questions_list"))

    @app.route("/questions/import", methods=["GET", "POST"])
    def questions_import():
        if request.method == "POST":
            f = request.files.get("file")
            if not f or not _allowed(f.filename, ALLOWED_XLSX_EXTENSIONS):
                flash("Envie um arquivo .xlsx ou .xls válido.", "danger")
                return redirect(request.url)
            fname = secure_filename(f.filename)
            path = os.path.join(UPLOAD_FOLDER, f"import_{uuid.uuid4().hex}_{fname}")
            f.save(path)
            try:
                imported = import_questions_from_xlsx(path)
                for data in imported:
                    errors = _validate_question_payload(**data)
                    if errors:
                        raise ValueError("; ".join(errors))
                    db.session.add(Question(**data))
                db.session.commit()
                flash(f"{len(imported)} pergunta(s) importada(s)!", "success")
            except Exception as e:
                db.session.rollback()
                app.logger.exception("Erro ao importar perguntas via XLSX")
                flash(f"Erro ao importar: {e}", "danger")
            finally:
                os.remove(path)
            return redirect(url_for("questions_list"))
        return render_template("questions_import.html")

    @app.route("/questions/import.json", methods=["POST"])
    @csrf.exempt
    def questions_import_json():
        try:
            if request.is_json:
                payload = request.get_json(silent=True) or {}
                imported = import_questions_from_json(io.StringIO(json.dumps(payload)))
            else:
                f = request.files.get("file")
                if not f or not _allowed(f.filename, ALLOWED_JSON_EXTENSIONS):
                    return {"error": "Envie um arquivo .json válido."}, 400
                imported = import_questions_from_json(f)

            created = 0
            for data in imported:
                errors = _validate_question_payload(**data)
                if errors:
                    return {"error": "; ".join(errors)}, 400
                db.session.add(Question(**data))
                created += 1
            db.session.commit()
            return {"created": created}, 201
        except Exception as e:
            db.session.rollback()
            app.logger.exception("Erro ao importar perguntas via JSON")
            return {"error": str(e)}, 400

    @app.route("/questions/export")
    def questions_export():
        questions = Question.query.all()
        buf = export_questions_to_xlsx(questions)
        return send_file(
            buf,
            mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            as_attachment=True,
            download_name="perguntas.xlsx",
        )

    @app.route("/questions/export.json")
    def questions_export_json():
        questions = Question.query.order_by(Question.created_at.asc()).all()
        payload = {
            "count": len(questions),
            "questions": [
                {
                    "id": q.id,
                    "text": q.text,
                    "answer_type": q.answer_type,
                    "correct_answer": q.correct_answer,
                    "custom_options": q.custom_options,
                    "options": q.get_options(),
                    "created_at": q.created_at.isoformat() if q.created_at else None,
                }
                for q in questions
            ],
        }
        return payload

    # ── Tests ────────────────────────────────────────────────────────────────

    @app.route("/tests")
    def tests_list():
        tests = Test.query.order_by(Test.created_at.desc()).all()
        return render_template("tests.html", tests=tests)

    @app.route("/tests/new", methods=["GET", "POST"])
    def test_new():
        questions = Question.query.order_by(Question.created_at).all()
        if request.method == "POST":
            title = request.form["title"].strip()
            if not title:
                flash("Título da prova é obrigatório.", "danger")
                return render_template("test_form.html", test=None, questions=questions), 400
            t = Test(
                title=title,
                description=request.form.get("description", "").strip() or None,
            )
            db.session.add(t)
            db.session.flush()
            q_ids = request.form.getlist("question_ids")
            for order, qid in enumerate(q_ids):
                db.session.add(TestQuestion(test_id=t.id, question_id=int(qid), order=order))
            db.session.commit()
            flash("Prova criada com sucesso!", "success")
            return redirect(url_for("tests_list"))
        return render_template("test_form.html", test=None, questions=questions)

    @app.route("/tests/<int:tid>/edit", methods=["GET", "POST"])
    def test_edit(tid):
        t = Test.query.get_or_404(tid)
        questions = Question.query.order_by(Question.created_at).all()
        if request.method == "POST":
            title = request.form["title"].strip()
            if not title:
                flash("Título da prova é obrigatório.", "danger")
                return render_template("test_form.html", test=t, questions=questions), 400
            t.title = title
            t.description = request.form.get("description", "").strip() or None
            TestQuestion.query.filter_by(test_id=t.id).delete()
            q_ids = request.form.getlist("question_ids")
            for order, qid in enumerate(q_ids):
                db.session.add(TestQuestion(test_id=t.id, question_id=int(qid), order=order))
            db.session.commit()
            flash("Prova atualizada!", "success")
            return redirect(url_for("tests_list"))
        return render_template("test_form.html", test=t, questions=questions)

    @app.route("/tests/<int:tid>/delete", methods=["POST"])
    def test_delete(tid):
        t = Test.query.get_or_404(tid)
        db.session.delete(t)
        db.session.commit()
        flash("Prova excluída.", "info")
        return redirect(url_for("tests_list"))

    @app.route("/tests/<int:tid>/pdf")
    def test_pdf(tid):
        t = Test.query.get_or_404(tid)
        out = os.path.join(UPLOAD_FOLDER, f"test_{tid}_{uuid.uuid4().hex}.pdf")
        try:
            generate_test_pdf(t, out)
            return send_file(out, mimetype="application/pdf", as_attachment=False,
                             download_name=f"prova_{tid}.pdf")
        finally:
            if os.path.exists(out):
                os.remove(out)

    # ── Scans ────────────────────────────────────────────────────────────────

    @app.route("/tests/<int:tid>/scans")
    def scans_list(tid):
        t = Test.query.get_or_404(tid)
        return render_template("scans.html", test=t)

    @app.route("/tests/<int:tid>/scans/new", methods=["GET", "POST"])
    def scan_new(tid):
        t = Test.query.get_or_404(tid)
        if request.method == "POST":
            name = request.form["respondent_name"].strip()
            if not name:
                flash("Nome do respondente é obrigatório.", "danger")
                return render_template("scan_form.html", test=t), 400
            obs = request.form.get("observation", "").strip() or None
            img_filename = None

            img_file = request.files.get("image")
            if img_file and img_file.filename and _allowed(img_file.filename, ALLOWED_SCAN_EXTENSIONS):
                safe = secure_filename(img_file.filename)
                img_filename = f"scan_{uuid.uuid4().hex}_{safe}"
                img_file.save(os.path.join(UPLOAD_FOLDER, img_filename))
                app.logger.info("Scan salvo: %s", img_filename)

            scan = Scan(test_id=tid, respondent_name=name, observation=obs, image_filename=img_filename)
            db.session.add(scan)
            db.session.flush()

            # Determine answers: manual form entries take priority; fall back to OMR
            auto_answers = {}
            if img_filename:
                try:
                    auto_answers, auto_confidences = process_scan_image(
                        os.path.join(UPLOAD_FOLDER, img_filename), t, with_confidence=True
                    )
                except Exception:
                    app.logger.exception("Falha no processamento OMR para scan %s", img_filename)
                    auto_answers = {}
                    auto_confidences = {}
            else:
                auto_confidences = {}

            for tq in t.questions:
                q = tq.question
                manual_key = f"answer_{q.id}"
                if manual_key in request.form and request.form[manual_key].strip():
                    answer = request.form[manual_key].strip()
                else:
                    answer = auto_answers.get(q.id)
                if answer:
                    confidence = None
                    if manual_key not in request.form or not request.form[manual_key].strip():
                        confidence = auto_confidences.get(q.id)
                    db.session.add(
                        Response(
                            scan_id=scan.id,
                            question_id=q.id,
                            answer=answer,
                            omr_confidence=confidence,
                        )
                    )

            db.session.commit()
            flash("Scan registrado com sucesso!", "success")
            return redirect(url_for("scan_report", tid=tid, sid=scan.id))

        return render_template("scan_form.html", test=t)

    @app.route("/tests/<int:tid>/scans/<int:sid>/report")
    def scan_report(tid, sid):
        t = Test.query.get_or_404(tid)
        scan = Scan.query.filter_by(id=sid, test_id=tid).first_or_404()
        correct, total = scan.calculate_score()
        return render_template("report.html", test=t, scan=scan, correct=correct, total=total)

    @app.route("/tests/<int:tid>/scans/<int:sid>/delete", methods=["POST"])
    def scan_delete(tid, sid):
        scan = Scan.query.filter_by(id=sid, test_id=tid).first_or_404()
        if scan.image_filename:
            path = os.path.join(UPLOAD_FOLDER, scan.image_filename)
            if os.path.exists(path):
                os.remove(path)
        db.session.delete(scan)
        db.session.commit()
        flash("Scan excluído.", "info")
        return redirect(url_for("scans_list", tid=tid))

    @app.route("/tests/<int:tid>/report")
    def test_report(tid):
        t = Test.query.get_or_404(tid)
        report_data = []
        for scan in t.scans:
            correct, total = scan.calculate_score()
            report_data.append({"scan": scan, "correct": correct, "total": total})
        return render_template("test_report.html", test=t, report_data=report_data)

    @app.route("/tests/<int:tid>/manifest.json")
    def test_manifest_json(tid):
        t = Test.query.get_or_404(tid)
        template = build_omr_template(t)
        manifest = {
            "version": "1.1",
            "template": template,
            "test": {
                "id": t.id,
                "title": t.title,
                "description": t.description,
                "created_at": t.created_at.isoformat() if t.created_at else None,
            },
            "questions": [],
        }
        for order, tq in enumerate(t.questions, start=1):
            q = tq.question
            manifest["questions"].append(
                {
                    "order": order,
                    "question_id": q.id,
                    "text": q.text,
                    "answer_type": q.answer_type,
                    "options": q.get_options(),
                    "correct_answer": q.correct_answer,
                }
            )
        return manifest

    @app.route("/tests/<int:tid>/results.json")
    def test_results_json(tid):
        t = Test.query.get_or_404(tid)
        result_rows = []
        for scan in t.scans:
            correct, total = scan.calculate_score()
            responses = {}
            for r in scan.responses:
                responses[str(r.question_id)] = r.answer
            result_rows.append(
                {
                    "scan_id": scan.id,
                    "respondent_name": scan.respondent_name,
                    "observation": scan.observation,
                    "created_at": scan.created_at.isoformat() if scan.created_at else None,
                    "score": {
                        "correct": correct,
                        "total": total,
                        "percent": round((correct / total) * 100, 2) if total else None,
                    },
                    "responses": responses,
                    "omr_confidence": {
                        str(r.question_id): r.omr_confidence
                        for r in scan.responses
                        if r.omr_confidence is not None
                    },
                    "image_filename": scan.image_filename,
                }
            )
        return {
            "version": "1.0",
            "test_id": t.id,
            "test_title": t.title,
            "scans_count": len(result_rows),
            "scans": result_rows,
        }

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename):
        safe_name = secure_filename(filename)
        if not safe_name or safe_name != filename:
            return ("Arquivo inválido.", 400)
        return send_from_directory(UPLOAD_FOLDER, safe_name)

    def _allowed(filename, allowed):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

    def _validate_question_payload(text, answer_type, correct_answer, custom_options):
        errors = []
        if not text:
            errors.append("Texto da pergunta é obrigatório.")
        if answer_type not in VALID_ANSWER_TYPES:
            errors.append("Tipo de resposta inválido.")
            return errors

        options = []
        if answer_type == "yes_no":
            options = ["Sim", "Não"]
        elif answer_type == "likert":
            options = ["1", "2", "3", "4", "5"]
        elif answer_type == "custom":
            options = [o.strip() for o in (custom_options or "").split(",") if o.strip()]
            if len(options) < 2:
                errors.append("Perguntas customizadas precisam de ao menos duas opções.")

        if correct_answer and options and correct_answer not in options:
            errors.append("Resposta correta deve ser uma opção válida da pergunta.")
        return errors

    def _ensure_sqlite_indexes(flask_app):
        db_uri = flask_app.config.get("SQLALCHEMY_DATABASE_URI", "")
        if not db_uri.startswith("sqlite"):
            return
        try:
            db.session.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_test_question ON test_questions (test_id, question_id)"
                )
            )
            db.session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_test_questions_test_id ON test_questions (test_id)"
                )
            )
            db.session.execute(
                text(
                    "CREATE UNIQUE INDEX IF NOT EXISTS uq_scan_question_response ON responses (scan_id, question_id)"
                )
            )
            db.session.execute(
                text(
                    "CREATE INDEX IF NOT EXISTS ix_responses_scan_id ON responses (scan_id)"
                )
            )
            columns = db.session.execute(text("PRAGMA table_info(responses)")).fetchall()
            col_names = {row[1] for row in columns}
            if "omr_confidence" not in col_names:
                db.session.execute(text("ALTER TABLE responses ADD COLUMN omr_confidence REAL"))
            db.session.commit()
        except Exception:
            db.session.rollback()
            flask_app.logger.exception(
                "Nao foi possivel garantir indexes no SQLite. Verifique dados duplicados existentes."
            )

    with app.app_context():
        _ensure_sqlite_indexes(app)

    return app


if __name__ == "__main__":
    app = create_app()
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=5000)
