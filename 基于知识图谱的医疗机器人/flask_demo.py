from flask import Flask,render_template,request
import jieba

app = Flask(__name__)

@app.route("/",methods=["GET","POST"])
def abc():
    if request.method == "GET":
        print("get")
        return render_template("index.html")
    else:
        print("post")
        inputtext = request.form.get("inputtext")

        r = model(inputtext)

        return render_template("index.html",data=" ".join(r))



if __name__ == "__main__":

    model = jieba.lcut
    app.run(host="127.0.0.1",port=9999,debug=True)
