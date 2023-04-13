from flask import Flask, render_template, request, make_response, render_template_string
from final_code import eng_to_python
import googletrans
from googletrans import Translator
from translator import translate_to_english

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/run", methods=["POST"])
def run():
    input_text = request.form["input_code"]
    translated_text = translate_to_english(input_text) 
    output_code = eng_to_python(translated_text)
    response = make_response(render_template_string("<h2>Output Code</h2><pre>{{ output_code }}</pre>", output_code=output_code))
    response.headers["Content-Disposition"] = "attachment; filename=output.py"
    response.headers["Content-type"] = "text/plain"
    return response

if __name__ == "__main__":
    app.run(debug=True)

# from flask import Flask, request, jsonify, render_template
# from final_code import eng_to_python
# app = Flask(__name__)

# @app.route("/")
# def index():
#     return render_template("index.html")



# @app.route('/run', methods=['POST'])
# def run():
#     input_code = request.json.get('input')
#     python_code = eng_to_python(input_code)

#     # Execute the Python code and get the output
#     output = ""
#     try:
#         exec(python_code)
#         output = str(eval("num"))
#     except Exception as e:
#         output = "Error: " + str(e)

#     return jsonify(output=output)

# if __name__ == '__main__':
#     app.run()

