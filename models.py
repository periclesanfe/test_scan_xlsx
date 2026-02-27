from flask_sqlalchemy import SQLAlchemy
from datetime import datetime

db = SQLAlchemy()


class Question(db.Model):
    __tablename__ = "questions"
    id = db.Column(db.Integer, primary_key=True)
    text = db.Column(db.String(1000), nullable=False)
    answer_type = db.Column(db.String(20), nullable=False)  # yes_no, likert, custom
    correct_answer = db.Column(db.String(200), nullable=True)
    custom_options = db.Column(db.String(500), nullable=True)  # comma-separated for custom type
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def get_options(self):
        if self.answer_type == "yes_no":
            return ["Sim", "Não"]
        elif self.answer_type == "likert":
            return ["1", "2", "3", "4", "5"]
        elif self.answer_type == "custom" and self.custom_options:
            return [o.strip() for o in self.custom_options.split(",") if o.strip()]
        return []


class Test(db.Model):
    __tablename__ = "tests"
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(200), nullable=False)
    description = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    questions = db.relationship("TestQuestion", backref="test", cascade="all, delete-orphan", order_by="TestQuestion.order")
    scans = db.relationship("Scan", backref="test", cascade="all, delete-orphan")


class TestQuestion(db.Model):
    __tablename__ = "test_questions"
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey("tests.id"), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey("questions.id"), nullable=False)
    order = db.Column(db.Integer, nullable=False, default=0)
    question = db.relationship("Question")


class Scan(db.Model):
    __tablename__ = "scans"
    id = db.Column(db.Integer, primary_key=True)
    test_id = db.Column(db.Integer, db.ForeignKey("tests.id"), nullable=False)
    respondent_name = db.Column(db.String(200), nullable=False)
    observation = db.Column(db.Text, nullable=True)
    image_filename = db.Column(db.String(300), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    responses = db.relationship("Response", backref="scan", cascade="all, delete-orphan")

    def calculate_score(self):
        correct = 0
        total = 0
        for resp in self.responses:
            tq = TestQuestion.query.filter_by(
                test_id=self.test_id, question_id=resp.question_id
            ).first()
            if tq and tq.question.correct_answer:
                total += 1
                if resp.answer == tq.question.correct_answer:
                    correct += 1
        return correct, total


class Response(db.Model):
    __tablename__ = "responses"
    id = db.Column(db.Integer, primary_key=True)
    scan_id = db.Column(db.Integer, db.ForeignKey("scans.id"), nullable=False)
    question_id = db.Column(db.Integer, db.ForeignKey("questions.id"), nullable=False)
    answer = db.Column(db.String(200), nullable=True)
    question = db.relationship("Question")
