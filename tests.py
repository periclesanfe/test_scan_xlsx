"""
Basic tests for the QuizScan application.
"""
import io
import os
import tempfile

import pytest

from app import create_app
from models import db as _db, Question, Test, TestQuestion, Scan, Response


@pytest.fixture
def app():
    tmp = tempfile.mkdtemp()
    app = create_app()
    app.config.update(
        TESTING=True,
        SQLALCHEMY_DATABASE_URI=f"sqlite:///{os.path.join(tmp, 'test.db')}",
        UPLOAD_FOLDER=tmp,
        WTF_CSRF_ENABLED=False,
    )
    with app.app_context():
        _db.create_all()
        yield app
        _db.session.remove()
        _db.drop_all()


@pytest.fixture
def client(app):
    return app.test_client()


# ── Question tests ───────────────────────────────────────────────────────────

def test_create_question_yes_no(client):
    rv = client.post("/questions/new", data={
        "text": "Você gostou?",
        "answer_type": "yes_no",
        "correct_answer": "Sim",
    }, follow_redirects=True)
    assert rv.status_code == 200
    assert b"Pergunta criada" in rv.data


def test_create_question_likert(client):
    rv = client.post("/questions/new", data={
        "text": "Avalie de 1 a 5",
        "answer_type": "likert",
        "correct_answer": "3",
    }, follow_redirects=True)
    assert rv.status_code == 200
    assert b"Pergunta criada" in rv.data


def test_create_question_custom(client):
    rv = client.post("/questions/new", data={
        "text": "Qual sua cor preferida?",
        "answer_type": "custom",
        "custom_options": "Azul, Verde, Vermelho",
        "correct_answer": "Azul",
    }, follow_redirects=True)
    assert rv.status_code == 200
    assert b"Pergunta criada" in rv.data


def test_question_options(app):
    with app.app_context():
        q_yn = Question(text="Q1", answer_type="yes_no")
        q_lk = Question(text="Q2", answer_type="likert")
        q_cu = Question(text="Q3", answer_type="custom", custom_options="A, B, C")
        assert q_yn.get_options() == ["Sim", "Não"]
        assert q_lk.get_options() == ["1", "2", "3", "4", "5"]
        assert q_cu.get_options() == ["A", "B", "C"]


def test_delete_question(client, app):
    with app.app_context():
        q = Question(text="Temp", answer_type="yes_no")
        _db.session.add(q)
        _db.session.commit()
        qid = q.id
    rv = client.post(f"/questions/{qid}/delete", follow_redirects=True)
    assert rv.status_code == 200
    assert b"exclu" in rv.data.lower()


# ── Test (prova) tests ────────────────────────────────────────────────────────

def test_create_test(client, app):
    with app.app_context():
        q = Question(text="Q?", answer_type="yes_no", correct_answer="Sim")
        _db.session.add(q)
        _db.session.commit()
        qid = q.id
    rv = client.post("/tests/new", data={
        "title": "Prova de exemplo",
        "description": "Descrição",
        "question_ids": [str(qid)],
    }, follow_redirects=True)
    assert rv.status_code == 200
    assert b"Prova criada" in rv.data


def test_test_pdf(client, app):
    with app.app_context():
        q = Question(text="Pergunta?", answer_type="yes_no")
        _db.session.add(q)
        _db.session.flush()
        t = Test(title="PDF Test")
        _db.session.add(t)
        _db.session.flush()
        _db.session.add(TestQuestion(test_id=t.id, question_id=q.id, order=0))
        _db.session.commit()
        tid = t.id
    rv = client.get(f"/tests/{tid}/pdf")
    assert rv.status_code == 200
    assert rv.content_type == "application/pdf"


# ── Scan / Report tests ───────────────────────────────────────────────────────

def test_create_scan(client, app):
    with app.app_context():
        q = Question(text="Q?", answer_type="yes_no", correct_answer="Sim")
        _db.session.add(q)
        _db.session.flush()
        t = Test(title="Prova Scan")
        _db.session.add(t)
        _db.session.flush()
        _db.session.add(TestQuestion(test_id=t.id, question_id=q.id, order=0))
        _db.session.commit()
        tid, qid = t.id, q.id

    rv = client.post(f"/tests/{tid}/scans/new", data={
        "respondent_name": "João",
        "observation": "Teste obs",
        f"answer_{qid}": "Sim",
    }, follow_redirects=True)
    assert rv.status_code == 200
    assert b"Jo" in rv.data


def test_scan_score(app):
    with app.app_context():
        q1 = Question(text="Q1", answer_type="yes_no", correct_answer="Sim")
        q2 = Question(text="Q2", answer_type="yes_no", correct_answer="Não")
        _db.session.add_all([q1, q2])
        _db.session.flush()
        t = Test(title="Score Test")
        _db.session.add(t)
        _db.session.flush()
        _db.session.add(TestQuestion(test_id=t.id, question_id=q1.id, order=0))
        _db.session.add(TestQuestion(test_id=t.id, question_id=q2.id, order=1))
        _db.session.flush()
        scan = Scan(test_id=t.id, respondent_name="Maria")
        _db.session.add(scan)
        _db.session.flush()
        _db.session.add(Response(scan_id=scan.id, question_id=q1.id, answer="Sim"))   # correct
        _db.session.add(Response(scan_id=scan.id, question_id=q2.id, answer="Sim"))   # wrong
        _db.session.commit()
        correct, total = scan.calculate_score()
        assert correct == 1
        assert total == 2


# ── XLSX import tests ─────────────────────────────────────────────────────────

def test_import_xlsx(client, app):
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.append(["text", "answer_type", "correct_answer", "custom_options"])
    ws.append(["Pergunta XLSX?", "yes_no", "Sim", ""])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    rv = client.post("/questions/import", data={
        "file": (buf, "questions.xlsx"),
    }, content_type="multipart/form-data", follow_redirects=True)
    assert rv.status_code == 200
    assert b"importada" in rv.data


def test_export_xlsx(client, app):
    with app.app_context():
        _db.session.add(Question(text="Export Q", answer_type="likert"))
        _db.session.commit()
    rv = client.get("/questions/export")
    assert rv.status_code == 200
    assert "spreadsheetml" in rv.content_type


# ── Security / validation tests ──────────────────────────────────────────────

def test_upload_path_traversal_blocked(client):
    rv = client.get("/uploads/../app.py")
    assert rv.status_code == 400


def test_scan_report_blocks_cross_test_access(client, app):
    with app.app_context():
        q = Question(text="Q?", answer_type="yes_no", correct_answer="Sim")
        _db.session.add(q)
        _db.session.flush()

        t1 = Test(title="T1")
        t2 = Test(title="T2")
        _db.session.add_all([t1, t2])
        _db.session.flush()

        _db.session.add(TestQuestion(test_id=t1.id, question_id=q.id, order=0))
        _db.session.add(TestQuestion(test_id=t2.id, question_id=q.id, order=0))
        _db.session.flush()

        scan = Scan(test_id=t1.id, respondent_name="Pessoa")
        _db.session.add(scan)
        _db.session.commit()
        t2_id, scan_id = t2.id, scan.id

    rv = client.get(f"/tests/{t2_id}/scans/{scan_id}/report")
    assert rv.status_code == 404


def test_create_question_rejects_invalid_correct_answer(client):
    rv = client.post("/questions/new", data={
        "text": "Questão inválida",
        "answer_type": "yes_no",
        "correct_answer": "Talvez",
    })
    assert rv.status_code == 400
    assert b"Resposta correta deve ser uma op" in rv.data


def test_import_questions_json_payload(client):
    rv = client.post("/questions/import.json", json={
        "questions": [
            {
                "text": "Pergunta JSON?",
                "answer_type": "yes_no",
                "correct_answer": "Sim",
                "custom_options": "",
            }
        ]
    })
    assert rv.status_code == 201
    assert rv.json["created"] == 1


def test_manifest_json_contains_template(client, app):
    with app.app_context():
        q = Question(text="Q Manifest", answer_type="yes_no", correct_answer="Sim")
        _db.session.add(q)
        _db.session.flush()
        t = Test(title="Manifest Test")
        _db.session.add(t)
        _db.session.flush()
        _db.session.add(TestQuestion(test_id=t.id, question_id=q.id, order=0))
        _db.session.commit()
        tid = t.id

    rv = client.get(f"/tests/{tid}/manifest.json")
    assert rv.status_code == 200
    assert rv.json["template"]["template_version"] == "1.1"
    assert rv.json["template"]["questions"][0]["question_id"] > 0


def test_results_json_includes_confidence_key(client, app):
    with app.app_context():
        q = Question(text="Q Results", answer_type="yes_no", correct_answer="Sim")
        _db.session.add(q)
        _db.session.flush()
        t = Test(title="Results Test")
        _db.session.add(t)
        _db.session.flush()
        _db.session.add(TestQuestion(test_id=t.id, question_id=q.id, order=0))
        _db.session.flush()
        scan = Scan(test_id=t.id, respondent_name="Ana")
        _db.session.add(scan)
        _db.session.flush()
        _db.session.add(Response(scan_id=scan.id, question_id=q.id, answer="Sim", omr_confidence=0.91))
        _db.session.commit()
        tid = t.id

    rv = client.get(f"/tests/{tid}/results.json")
    assert rv.status_code == 200
    assert "omr_confidence" in rv.json["scans"][0]
