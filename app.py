import os
from flask import Flask, render_template, request, redirect, url_for,Blueprint, make_response, render_template, flash, session, g
from flask_login import LoginManager, login_user, logout_user, current_user, login_required
from flask_sqlalchemy import SQLAlchemy
import numpy as np
import pandas as pd
import pylab as pl
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA as sklearnPCA
from sklearn.decomposition import PCA
from sqlalchemy import create_engine, MetaData
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.cluster import SpectralClustering
from sklearn import metrics
from sklearn.metrics import pairwise_distances
from sklearn.metrics import davies_bouldin_score

app = Flask(__name__)
DATABASE_URL = os.environ.get('DATABASE_URL', 'sqlite:////tmp/flask_app.db')
engine = create_engine(DATABASE_URL)
app.config['SQLALCHEMY_DATABASE_URI'] = DATABASE_URL
app.config['SQLALCHEMY_TRACK_MODIFICATIONS']=False
app.config['SECRET_KEY']='fightclub1999'
db = SQLAlchemy(app)


class Ques(db.Model):
 id = db.Column(db.Integer, primary_key = True)
 ins = db.Column(db.String(80))
 orgname = db.Column(db.String(80))
 address = db.Column(db.String(80))
 q1 = db.Column(db.String(480))
 q2 = db.Column(db.String(480))
 q3 = db.Column(db.String(480))
 q4 = db.Column(db.String(480))
 q5 = db.Column(db.String(480))

class User(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    username = db.Column(db.String(120), unique = True)
    pw_hash = db.Column(db.String(480))
    name = db.Column(db.String(240))     
    def is_authenticated(self):
        return True
    def is_active(self):
        return True
    def is_anonymous(self):
        return False
    def get_id(self):
        return self.id
    def __repr__(self):
        return '<User %r>' % (self.nickname)

class Res(db.Model):
 id = db.Column(db.Integer, primary_key = True)
 q1=db.Column(db.Float(8))
 q2=db.Column(db.Float(8))
 q3=db.Column(db.Float(8))
 q4=db.Column(db.Float(8))
 q5=db.Column(db.Float(8))

class Token(db.Model):
 id = db.Column(db.Integer, primary_key = True)
 uid = db.Column(db.Integer)
 tok= db.Column(db.Integer)
 check = db.Column(db.Integer)


class Dataset(db.Model):
 id = db.Column(db.Integer, primary_key = True)
 uid = db.Column(db.Integer)
 orgid = db.Column(db.Integer)
 q1=db.Column(db.Float(8))
 q2=db.Column(db.Float(8))
 q3=db.Column(db.Float(8))
 q4=db.Column(db.Float(8))
 q5=db.Column(db.Float(8))


login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load user into each response
@login_manager.user_loader
def load_user(id):
        return User.query.get(int(id))


@app.before_request
def before_request():
   g.user = current_user

@app.route('/test', methods=['GET', 'POST'])
def test():
 APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
 APP_STATIC = os.path.join(APP_ROOT, 'static')
 f = open(os.path.join(APP_STATIC, 'corr.csv'), "r")
 f.readline()
 for x in f:
  y = x.split(",")
  data = Dataset(uid=int(y[0]), orgid=int(y[1]), q1=float(y[2]), q2=float(y[3]), q3=float(y[4]),q4=float(y[5]),q5=float(y[6]) )
  db.session.add(data)
  db.session.commit()
 f.close()
 return "hell"

@app.route('/test3', methods=['GET', 'POST'])
def test3():
 ques = Ques(orgname="Sonali Bank", div="Management", address="KUET, Khulna", q1="this is q1?", q2="this is q1?",
 q3="this is q3?",q4="this is q4?",q5="this is q5?")
 db.session.add(ques)
 db.session.commit()
 ques = Ques(orgname="Janata Bank", div="Management", address="KUET, Khulna", q1="this is q1?", q2="this is q1?",
 q3="this is q3?",q4="this is q4?",q5="this is q5?")
 db.session.add(ques)
 db.session.commit()
 ques = Ques(orgname="Sonali Bank", div="Accounting", address="KUET, Khulna", q1="this is q1?", q2="this is q1?",
 q3="this is q3?",q4="this is q4?",q5="this is q5?")
 db.session.add(ques)
 db.session.commit()
 ques = Ques(orgname="Janata Bank", div="Accounting", address="KUET, Khulna", q1="this is q1?", q2="this is q1?",
 q3="this is q3?",q4="this is q4?",q5="this is q5?")
 db.session.add(ques)
 db.session.commit()
 return "hell"

@app.route('/test4', methods=['GET', 'POST'])
def test4():
 APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
 APP_STATIC = os.path.join(APP_ROOT, 'static')
 f = open(os.path.join(APP_STATIC, 'zaberques.csv'), "r")
 f.readline()
 for x in f:
  y = x.split(",")
  ques = Ques(id=int(y[0]), ins=(y[1]),orgname=(y[2]),address=(y[3]), q1=(y[4]), q2=(y[5]), q3=(y[6]),q4=(y[7]),q5=(y[8]) )
  db.session.add(ques)
  db.session.commit()
 f.close()
 return "hell4"

@app.route('/test5', methods=['GET', 'POST'])
def test5():
 APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
 APP_STATIC = os.path.join(APP_ROOT, 'static')
 f = open(os.path.join(APP_STATIC, 'token.csv'), "r")
 f.readline()
 for x in f:
  y = x.split(",")
  token = Token(uid=int(y[0]), tok=int(y[1]), check=float(y[2]) )
  db.session.add(token)
  db.session.commit()
 f.close()
 return "hell5"



@app.route('/test2', methods=['GET', 'POST'])
def test2():
 if request.method == 'POST':
  if request.form['btn'] == 'ADMIN PANEL':
   return redirect(url_for('admin_panel'))
  if request.form['btn'] == 'LOG OUT':
   logout_user()
   return redirect(url_for('index'))
 APP_ROOT = os.path.dirname(os.path.abspath(__file__))   # refers to application_top
 APP_STATIC = os.path.join(APP_ROOT, 'static')
 df = pd.read_sql_table("dataset",con=engine)
 df.dropna(inplace=True)
 df=df.drop('id',axis=1)
 df=df.drop('uid',axis=1)
 df=df.groupby(['orgid']).mean()
 from sklearn import preprocessing
 minmax_processed = preprocessing.MinMaxScaler().fit_transform(df)
 df_numeric_scaled = pd.DataFrame(minmax_processed, index=df.index, columns=df.columns)
 kmeans = KMeans(n_clusters=3)
 kmeans.fit(df_numeric_scaled)

 df2=df
 df['cluster'] = kmeans.labels_
 measure={}
 measure['0']=metrics.silhouette_score(df_numeric_scaled,df['cluster'], metric='euclidean')
 measure['1']=metrics.calinski_harabasz_score(df_numeric_scaled, df['cluster'])
 measure['2']=davies_bouldin_score(df_numeric_scaled, df['cluster'])

 df['total'] =df['q1']+ df['q2']+df['q3']+ df['q4']+df['q5']
 res=df.groupby(['cluster']).mean()
 res=res.sort_values(by=['total'],ascending = False)
 res['cluster']=res.index.values
 for x in range(0,df.shape[0]):
    if df.iloc[x,5] == res.iloc[0,6]:
        df.iloc[x,5]=0
    elif df.iloc[x,5] == res.iloc[1,6]:
        df.iloc[x,5]=1
    elif df.iloc[x,5] == res.iloc[2,6]:
        df.iloc[x,5]=2 
 t=df['cluster']
 X = df.iloc[:,0:5].values
 Y=df['cluster']
 from sklearn.preprocessing import StandardScaler
 X_std = StandardScaler().fit_transform(X)
 from sklearn.decomposition import PCA as sklearnPCA
 sklearn_pca = sklearnPCA(n_components=2)
 Y_sklearn = sklearn_pca.fit_transform(X_std)
 #t2=Y_sklearn
 plt.clf()
 plt.cla()
 plt.close()
 labelname = {0: 'Good',
              1: 'Mediocore',
              2: 'Corrupt'}
 with plt.style.context('seaborn-whitegrid'):
   plt.figure(figsize=(6,4)) 
   for lab, col in zip((0, 1, 2),
                        ('blue', 'green', 'red')):
        plt.scatter(Y_sklearn[Y==lab, 0],
                    Y_sklearn[Y==lab, 1],
                    label=labelname[lab],
                    c=col)
   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')
   plt.legend(loc='upper right')
   now = datetime.now()
   date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
   date_time=(date_time.translate({ord(' '): None}))
   date_time=(date_time.translate({ord(','): None}))
   date_time=(date_time.translate({ord('/'): None}))
   date_time=(date_time.translate({ord(':'): None}))
   imgname=str(date_time)

   imgname=imgname+'.png'
   plt.savefig(os.path.join(APP_STATIC, imgname))
 
 plt.clf()

 plt.cla()
 plt.close()
 plt.figure(figsize=(6,4)) 
 clrs=['blue', 'green', 'red']
 axis = sns.barplot(x=['Good', 'Mediocore', 'Bad'],y=df.groupby(['cluster']).count()['q1'].values,palette=clrs)
 x=axis.set_xlabel("Cluster Class")
 x=axis.set_ylabel("Number of Orgnization")
 imgname2=str(date_time)+'1'+'.png'
 plt.savefig(os.path.join(APP_STATIC, imgname2))
 ##end of K-Means

 from sklearn_extra.cluster import KMedoids
 kmedoids = KMedoids(n_clusters=3).fit(df_numeric_scaled)
 df=df2
 df['cluster'] = kmedoids.labels_
 measure['3']=metrics.silhouette_score(df_numeric_scaled,df['cluster'], metric='euclidean')
 measure['4']=metrics.calinski_harabasz_score(df_numeric_scaled, df['cluster'])
 measure['5']=davies_bouldin_score(df_numeric_scaled, df['cluster'])
 df['total'] =df['q1']+ df['q2']+df['q3']+ df['q4']+df['q5']
 res=df.groupby(['cluster']).mean()
 res=res.sort_values(by=['total'],ascending = False)
 res['cluster']=res.index.values
 for x in range(0,df.shape[0]):
    if df.iloc[x,5] == res.iloc[0,6]:
        df.iloc[x,5]=0
    elif df.iloc[x,5] == res.iloc[1,6]:
        df.iloc[x,5]=1
    elif df.iloc[x,5] == res.iloc[2,6]:
        df.iloc[x,5]=2 
 t=df['cluster']
 X = df.iloc[:,0:5].values
 Y=df['cluster']
 from sklearn.preprocessing import StandardScaler
 X_std = StandardScaler().fit_transform(X)
 from sklearn.decomposition import PCA as sklearnPCA
 sklearn_pca = sklearnPCA(n_components=2)
 Y_sklearn = sklearn_pca.fit_transform(X_std)
 #t2=Y_sklearn
 plt.clf()
 plt.cla()
 plt.close()
 plt.figure(figsize=(6,4)) 
 with plt.style.context('seaborn-whitegrid'):
   plt.figure(figsize=(6,4)) 
   for lab, col in zip((0, 1, 2),
                        ('blue', 'green', 'red')):
        plt.scatter(Y_sklearn[Y==lab, 0],
                    Y_sklearn[Y==lab, 1],
                    label=labelname[lab],
                    c=col)
   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')
   plt.legend(loc='upper right')

   imgname3=date_time+'kmd'+'.png'
   plt.savefig(os.path.join(APP_STATIC, imgname3))
 
 plt.clf()

 plt.cla()
 plt.close()
 plt.figure(figsize=(6,4)) 
 axis = sns.barplot(x=['Good', 'Mediocore', 'Bad'],y=df.groupby(['cluster']).count()['q1'].values,palette=clrs)
 x=axis.set_xlabel("Cluster Class")
 x=axis.set_ylabel("Number of Orgnization")
 imgname4=str(date_time)+'kmd'+'1'+'.png'
 plt.savefig(os.path.join(APP_STATIC, imgname4))
 #end of k mediod
 kmedoids = SpectralClustering(n_clusters=3).fit(df_numeric_scaled)
 df=df2
 df['cluster'] = kmedoids.labels_
 #measure db< better separation between , cl> better defined  clsuters,
 # sill >, distance between ppints of two clusters
 
 measure['6']=metrics.silhouette_score(df_numeric_scaled,df['cluster'], metric='euclidean')
 measure['7']=metrics.calinski_harabasz_score(df_numeric_scaled, df['cluster'])
 measure['8']=davies_bouldin_score(df_numeric_scaled, df['cluster'])
 df['total'] =df['q1']+ df['q2']+df['q3']+ df['q4']+df['q5']
 res=df.groupby(['cluster']).mean()
 res=res.sort_values(by=['total'],ascending = False)
 res['cluster']=res.index.values
 for x in range(0,df.shape[0]):
    if df.iloc[x,5] == res.iloc[0,6]:
        df.iloc[x,5]=0
    elif df.iloc[x,5] == res.iloc[1,6]:
        df.iloc[x,5]=1
    elif df.iloc[x,5] == res.iloc[2,6]:
        df.iloc[x,5]=2 
 t=df['cluster']
 X = df.iloc[:,0:5].values
 Y=df['cluster']
 from sklearn.preprocessing import StandardScaler
 X_std = StandardScaler().fit_transform(X)
 from sklearn.decomposition import PCA as sklearnPCA
 sklearn_pca = sklearnPCA(n_components=2)
 Y_sklearn = sklearn_pca.fit_transform(X_std)
 #t2=Y_sklearn
 plt.clf()
 plt.cla()
 plt.close()
 with plt.style.context('seaborn-whitegrid'):
   plt.figure(figsize=(6,4)) 
   for lab, col in zip((0, 1, 2),
                        ('blue', 'green', 'red')):
        plt.scatter(Y_sklearn[Y==lab, 0],
                    Y_sklearn[Y==lab, 1],
                    label=labelname[lab],
                    c=col)
   plt.xlabel('Principal Component 1')
   plt.ylabel('Principal Component 2')
   plt.legend(loc='upper right')

   imgname5=date_time+'sp'+'.png'
   plt.savefig(os.path.join(APP_STATIC, imgname5))
 
 plt.clf()

 plt.cla()
 plt.close()
 axis = sns.barplot(x=['Good', 'Mediocore', 'Bad'],y=df.groupby(['cluster']).count()['q1'].values,palette=clrs)
 x=axis.set_xlabel("Cluster Class")
 x=axis.set_ylabel("Number of Orgnization")
 imgname6=str(date_time)+'sp'+'1'+'.png'
 plt.savefig(os.path.join(APP_STATIC, imgname6))
 
 
 
 return render_template('test2.html',img1='/static/'+imgname, img2='/static/'+imgname2, img3='/static/'+imgname3, 
 						  img4='/static/'+imgname4, img5='/static/'+imgname5, img6='/static/'+imgname6,measure=measure)

@app.route('/', methods=['GET', 'POST'])
def index():
  if request.method == 'POST':
    if request.form['btn'] == 'HOME':
      return redirect(url_for('index'))
    if request.form['btn'] == 'LOG IN':
      return redirect(url_for('login'))
    if request.form['btn'] == 'SIGN UP':
      return redirect(url_for('signup'))
  if g.user is not None and g.user.is_authenticated:
   return redirect(url_for('orglist'))
  return render_template('index.html')

@app.route('/response', methods=['GET', 'POST'])
@login_required
def response():
 if request.method == 'POST':
  if request.form['btn'] == 'EXPLORE':
   return redirect(url_for('orglist'))
  if request.form['btn'] == 'LOG OUT':
   logout_user()
   return redirect(url_for('index'))
 return render_template('response.html')

@app.route('/badresponse', methods=['GET', 'POST'])
@login_required
def badresponse():
 if request.method == 'POST':
  if request.form['btn'] == 'EXPLORE':
   return redirect(url_for('orglist'))
  if request.form['btn'] == 'LOG OUT':
   logout_user()
   return redirect(url_for('index'))
 return render_template('badresponse.html')




@app.route('/admin_panel', methods=['GET', 'POST'])
@login_required
def admin_panel():
 return render_template('admin_panel.html')


@app.route('/result', methods=['GET', 'POST'])
@login_required
def result():
 now = datetime.now()
 date_time = now.strftime("%m/%d/%Y, %H:%M:%S")
 date_time=(date_time.translate({ord(' '): None}))
 date_time=(date_time.translate({ord(','): None}))
 date_time=(date_time.translate({ord('/'): None}))
 date_time=(date_time.translate({ord(':'): None}))
 list_of_files=str(date_time)
 return list_of_files

@app.route('/orglist', methods=['GET', 'POST'])
@login_required
def orglist():
 if request.method == 'POST':
  if request.form['btn'] == 'EXPLORE':
   return redirect(url_for('orglist'))
  if request.form['btn'] == 'LOG OUT':
   logout_user()
   return redirect(url_for('index'))
 ques=Ques.query.distinct(Ques.ins)
 ques1={}
 ques2={}
 ques3={}
 for val in Ques.query.all():
  if val.ins in ques1:
   ques1[val.ins].append(val.id)
   ques2[val.id]=val.orgname
   ques3[val.id]=val.address
  else:
   ques1[val.ins]=[val.id]
   ques2[val.id]=val.orgname
   ques3[val.id]=val.address
#ques1[ins->all id], q2,3= id->name,arddrss
 return render_template('org_list.html', ques=ques, ques1=ques1, ques2=ques2, ques3=ques3)

@app.route('/quespage/<ins>/<orgname>/<orgid>', methods=['GET','POST'])
@login_required
def quespage(ins,orgname, orgid):
 if request.method == 'POST':
  if request.form['btn'] == 'EXPLORE':
   return redirect(url_for('orglist'))
  if request.form['btn'] == 'LOG OUT':
   logout_user()
   return redirect(url_for('index'))
  if request.form['btn'] == 'Submit':
   uid=current_user.get_id()
   q1=request.form['q1']
   q2=request.form['q2']
   q3=request.form['q3']
   q4=request.form['q4']
   q5=request.form['q5']
   tok=request.form['tok']
   token = Token.query.filter(Token.uid==g.user.id,Token.tok==tok).first()
   if token is None or token.check==1:
    return redirect(url_for('badresponse')) 
   else:
    session.query(Token).filter(Token.uid==g.user.id, Token.tok=tok).update({Token.check: 1})
    session.commit() 
    dataset = Dataset(uid=uid,orgid=orgid,q1=q1, q2=q2, q3=q3,q4=q4,q5=q5)
    db.session.add(dataset)
    db.session.commit()
    return redirect(url_for('response'))	
 questions={}
 ques=Ques.query.filter(Ques.ins==ins, Ques.orgname==orgname,Ques.id==orgid).first()
 questions['1']=ques.q1
 questions['2']=ques.q2
 questions['3']=ques.q3
 questions['4']=ques.q4
 questions['5']=ques.q5
 return render_template('quespage.html', questions=questions, orgname=ins, div=orgname)


@app.route('/login', methods=['GET', 'POST'])
def login():
  if request.method == 'POST':
    if request.form['btn'] == 'HOME':
      return redirect(url_for('index'))
    if request.form['btn'] == 'LOG IN':
      return redirect(url_for('login'))
    if request.form['btn'] == 'SIGN UP':
      return redirect(url_for('signup'))
    if request.form['btn'] == 'submit':
      username = request.form['username']
      user = User.query.filter(User.username==username).first()
      if user is None:
          flash('No such user. Please try again')
          return render_template('login.html')
      if user.pw_hash != request.form['password']:
          flash('Incorrect password. Please try again')
          return render_template('login.html')
      login_user(user)
      flash("Logged in successfully")
      if(username=='admin@sys.com'):
      	return redirect(url_for('admin_panel'))
      else:
      	return redirect(url_for('orglist'))

  if g.user is not None and g.user.is_authenticated:
   if(g.user.username == 'admin@sys.com'):
    return redirect(url_for('admin_panel'))
   else:
   	return redirect(url_for('orglist'))
         

  return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():   
  if request.method == 'POST':
    if request.form['btn'] == 'HOME':
      return redirect(url_for('index'))
    if request.form['btn'] == 'LOG IN':
      return redirect(url_for('login'))
    if request.form['btn'] == 'SIGN UP':
      return redirect(url_for('signup'))
    if request.form['btn'] == 'submit':
        username = request.form['email']
        if User.query.filter(User.username==username).first():
            flash('User already exists. Please log in.')
            return redirect(url_for('login'))
        pw_hash = request.form['password']
        user = User(username=username, pw_hash=pw_hash)
        db.session.add(user)
        db.session.commit()
        flash('User successfully registered. Please log in.')
        return redirect(url_for('login'))
  return render_template('signup.html')



if __name__ == '__main__':
  db.create_all()
  app.run(debug=True)
