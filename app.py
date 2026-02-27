import os
import uuid

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify
from werkzeug.utils import secure_filename

from models import db, Question, Test, TestQuestion, Scan, Response
from utils import (
    generate_test_pdf,
    process_scan_image,
    import_questions_from_xlsx,
    export_questions_to_xlsx,
)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), "uploads")
ALLOWED_SCAN_EXTENSIONS = {"png", "jpg", "jpeg", "gif", "bmp", "tiff", "pdf"}
ALLOWED_XLSX_EXTENSIONS = {"xlsx", "xls"}


def create_app():
    app = Flask(__name__)
    app.config["SECRET_KEY"] = os.environ.get("SECRET_KEY", "dev-secret-key")
    app.config["SQLALCHEMY_DATABASE_URI"] = os.environ.get(
        "DATABASE_URL", "sqlite:///quiz.db"
    )
    app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False
    app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
    app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # 32 MB

    os.makedirs(UPLOAD_FOLDER, exist_ok=True)

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
            q = Question(
                text=request.form["text"].strip(),
                answer_type=request.form["answer_type"],
                correct_answer=request.form.get("correct_answer", "").strip() or None,
                custom_options=request.form.get("custom_options", "").strip() or None,
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
            q.text = request.form["text"].strip()
            q.answer_type = request.form["answer_type"]
            q.correct_answer = request.form.get("correct_answer", "").strip() or None
            q.custom_options = request.form.get("custom_options", "").strip() or None
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
                    db.session.add(Question(**data))
                db.session.commit()
                flash(f"{len(imported)} pergunta(s) importada(s)!", "success")
            except Exception as e:
                flash(f"Erro ao importar: {e}", "danger")
            finally:
                os.remove(path)
            return redirect(url_for("questions_list"))
        return render_template("questions_import.html")

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

    # ── Tests ────────────────────────────────────────────────────────────────

    @app.route("/tests")
    def tests_list():
        tests = Test.query.order_by(Test.created_at.desc()).all()
        return render_template("tests.html", tests=tests)

    @app.route("/tests/new", methods=["GET", "POST"])
    def test_new():
        questions = Question.query.order_by(Question.created_at).all()
        if request.method == "POST":
            t = Test(
                title=request.form["title"].strip(),
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
            t.title = request.form["title"].strip()
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
            obs = request.form.get("observation", "").strip() or None
            img_filename = None

            img_file = request.files.get("image")
            if img_file and img_file.filename and _allowed(img_file.filename, ALLOWED_SCAN_EXTENSIONS):
                safe = secure_filename(img_file.filename)
                img_filename = f"scan_{uuid.uuid4().hex}_{safe}"
                img_file.save(os.path.join(UPLOAD_FOLDER, img_filename))

            scan = Scan(test_id=tid, respondent_name=name, observation=obs, image_filename=img_filename)
            db.session.add(scan)
            db.session.flush()

            # Determine answers: manual form entries take priority; fall back to OMR
            auto_answers = {}
            if img_filename:
                try:
                    auto_answers = process_scan_image(
                        os.path.join(UPLOAD_FOLDER, img_filename), t
                    )
                except Exception:
                    auto_answers = {}

            for tq in t.questions:
                q = tq.question
                manual_key = f"answer_{q.id}"
                if manual_key in request.form and request.form[manual_key].strip():
                    answer = request.form[manual_key].strip()
                else:
                    answer = auto_answers.get(q.id)
                if answer:
                    db.session.add(Response(scan_id=scan.id, question_id=q.id, answer=answer))

            db.session.commit()
            flash("Scan registrado com sucesso!", "success")
            return redirect(url_for("scan_report", tid=tid, sid=scan.id))

        return render_template("scan_form.html", test=t)

    @app.route("/tests/<int:tid>/scans/<int:sid>/report")
    def scan_report(tid, sid):
        t = Test.query.get_or_404(tid)
        scan = Scan.query.get_or_404(sid)
        correct, total = scan.calculate_score()
        return render_template("report.html", test=t, scan=scan, correct=correct, total=total)

    @app.route("/tests/<int:tid>/scans/<int:sid>/delete", methods=["POST"])
    def scan_delete(tid, sid):
        scan = Scan.query.get_or_404(sid)
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

    @app.route("/uploads/<path:filename>")
    def uploaded_file(filename):
        return send_file(os.path.join(UPLOAD_FOLDER, filename))

    def _allowed(filename, allowed):
        return "." in filename and filename.rsplit(".", 1)[1].lower() in allowed

    return app


if __name__ == "__main__":
    app = create_app()
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(debug=debug, port=5000)
