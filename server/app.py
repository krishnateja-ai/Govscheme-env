"""
app.py

FastAPI server for GovScheme-Env.
Exposes all OpenEnv-required HTTP endpoints.

Endpoints:
  GET  /        -- interactive demo landing page
  POST /reset   -- start a new episode
  POST /step    -- take one action
  GET  /state   -- see internal episode state
  GET  /tasks   -- list all 3 tasks
  GET  /health  -- liveness check (used by HF Spaces and validator)
  GET  /schemes -- browse all 18 schemes
"""
from __future__ import annotations

import dataclasses
import os
import sys
from typing import Any, Dict, List, Optional

sys.path.insert(0, os.path.dirname(__file__))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel

from govscheme_environment import GovSchemeEnvironment
from models import GovSchemeAction


app = FastAPI(
    title="GovScheme-Env",
    description=(
        "OpenEnv environment: Government Scheme Eligibility Matching. "
        "An AI agent helps Indian citizens find, rank, and apply for welfare schemes."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_env = GovSchemeEnvironment()


class ResetRequest(BaseModel):
    task_name: str = "scheme_identification"
    citizen_id: Optional[str] = None
    seed: Optional[int] = None

class StepRequest(BaseModel):
    action_type: str
    scheme_ids: Optional[List[str]] = None
    ranked_schemes: Optional[List[Dict[str, Any]]] = None
    form_data: Optional[Dict[str, Any]] = None
    reasoning: Optional[str] = None


def _obs_to_dict(obs) -> Dict:
    return dataclasses.asdict(obs)

def _state_to_dict(state) -> Dict:
    return dataclasses.asdict(state)


@app.get("/", response_class=HTMLResponse)
def root():
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>GovScheme-Env - OpenEnv</title>
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:system-ui,-apple-system,sans-serif;background:#f4f6f8;color:#1a1a1a;line-height:1.5}
.page{max-width:980px;margin:0 auto;padding:28px 16px 48px}
.hero{text-align:center;padding:36px 0 28px}
.tag{display:inline-block;background:#E1F5EE;color:#0F6E56;font-size:11px;font-weight:600;padding:4px 12px;border-radius:99px;letter-spacing:.04em;margin-bottom:14px;text-transform:uppercase}
.hero h1{font-size:26px;font-weight:700;color:#111;margin-bottom:8px}
.hero p{font-size:14px;color:#666;max-width:520px;margin:0 auto 24px;line-height:1.65}
.stat-row{display:flex;justify-content:center;gap:12px;flex-wrap:wrap;margin-bottom:32px}
.stat{background:#fff;border:0.5px solid #e2e2e2;border-radius:10px;padding:14px 20px;text-align:center;min-width:110px}
.stat-num{font-size:24px;font-weight:700;color:#1D9E75}
.stat-lbl{font-size:11px;color:#888;margin-top:2px}
.main-grid{display:grid;grid-template-columns:420px 1fr;gap:18px;align-items:start}
@media(max-width:780px){.main-grid{grid-template-columns:1fr}}
.card{background:#fff;border:0.5px solid #e2e2e2;border-radius:13px;padding:20px}
.card-title{font-size:13px;font-weight:700;color:#111;margin-bottom:16px;padding-bottom:10px;border-bottom:0.5px solid #f0f0f0;letter-spacing:.01em}
.field{margin-bottom:11px}
.field label{display:block;font-size:11px;font-weight:600;color:#777;margin-bottom:4px;text-transform:uppercase;letter-spacing:.04em}
.field input,.field select{width:100%;padding:8px 10px;border:0.5px solid #ddd;border-radius:7px;font-size:13px;color:#1a1a1a;background:#fafafa;outline:none;transition:border-color .15s}
.field input:focus,.field select:focus{border-color:#1D9E75}
.row2{display:grid;grid-template-columns:1fr 1fr;gap:10px}
.checks-grid{display:grid;grid-template-columns:1fr 1fr;gap:7px;margin-top:4px}
.check-item{display:flex;align-items:center;gap:7px;font-size:12px;color:#444;cursor:pointer;padding:6px 8px;border:0.5px solid #eee;border-radius:7px;transition:background .1s}
.check-item:hover{background:#f7f7f7}
.check-item input[type=checkbox]{width:14px;height:14px;accent-color:#1D9E75;cursor:pointer}
.check-item.checked{background:#E1F5EE;border-color:#9FE1CB;color:#085041}
.presets{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:14px}
.preset-btn{padding:5px 12px;border:0.5px solid #ddd;border-radius:99px;font-size:11px;font-weight:600;cursor:pointer;background:#fff;color:#555;transition:all .15s}
.preset-btn:hover{background:#f0f0f0;border-color:#bbb}
.preset-btn.active{background:#1D9E75;color:#fff;border-color:#1D9E75}
.find-btn{width:100%;padding:12px;background:#1D9E75;color:#fff;border:none;border-radius:9px;font-size:14px;font-weight:700;cursor:pointer;margin-top:12px;letter-spacing:.02em;transition:background .15s}
.find-btn:hover{background:#0F6E56}
.find-btn:disabled{background:#9FE1CB;cursor:not-allowed}
.results-placeholder{padding:48px 20px;text-align:center;color:#aaa;font-size:13px}
.dot-row{display:flex;justify-content:center;gap:6px;margin-bottom:10px}
.dot{width:8px;height:8px;border-radius:50%;background:#e0e0e0}
.result-header{margin-bottom:14px}
.result-count{font-size:20px;font-weight:700;color:#111}
.result-sub{font-size:12px;color:#888;margin-top:2px}
.pill-row{display:flex;gap:6px;flex-wrap:wrap;margin-top:8px}
.pill{font-size:11px;font-weight:600;padding:3px 10px;border-radius:99px}
.pg{background:#EAF3DE;color:#27500A}
.pb{background:#E6F1FB;color:#185FA5}
.scheme-card{background:#fff;border:0.5px solid #e8e8e8;border-radius:10px;padding:14px 15px;margin-bottom:9px;transition:border-color .15s}
.scheme-card:hover{border-color:#9FE1CB}
.scheme-card.rank1{border-color:#5DCAA5;background:#fafffe}
.sc-top{display:flex;justify-content:space-between;align-items:flex-start;gap:8px;margin-bottom:4px}
.sc-name{font-size:13px;font-weight:700;color:#111;display:flex;align-items:center;gap:7px}
.rn{display:inline-flex;width:20px;height:20px;border-radius:50%;background:#E6F1FB;color:#185FA5;font-size:10px;font-weight:700;align-items:center;justify-content:center;flex-shrink:0}
.rn.gold{background:#FAEEDA;color:#633806}
.benefit-pill{font-size:11px;font-weight:600;background:#EAF3DE;color:#27500A;padding:3px 9px;border-radius:99px;white-space:nowrap;flex-shrink:0}
.benefit-pill.access{background:#E6F1FB;color:#185FA5}
.sc-ministry{font-size:11px;color:#999;margin-bottom:5px}
.sc-desc{font-size:12px;color:#555;line-height:1.55}
.sc-bar{display:flex;align-items:center;gap:8px;margin-top:8px}
.bar-track{flex:1;height:4px;background:#f0f0f0;border-radius:2px}
.bar-fill{height:4px;border-radius:2px;transition:width .6s ease}
.bar-label{font-size:10px;color:#aaa;min-width:28px;text-align:right;font-weight:600}
.spinner{display:flex;gap:5px;justify-content:center;padding:40px 0}
.spin-dot{width:8px;height:8px;border-radius:50%;background:#1D9E75;animation:bounce .9s ease-in-out infinite}
.spin-dot:nth-child(2){animation-delay:.15s}
.spin-dot:nth-child(3){animation-delay:.3s}
@keyframes bounce{0%,60%,100%{transform:translateY(0)}30%{transform:translateY(-8px)}}
</style>
</head>
<body>
<div class="page">
<div class="hero">
  <div class="tag">OpenEnv - Real-world benchmark</div>
  <h1>GovScheme-Env</h1>
  <p>Enter any citizen profile to instantly see all Indian government welfare schemes they qualify for, ranked by benefit value. Built as an AI agent training environment for the Meta x HuggingFace OpenEnv hackathon.</p>
  <div class="stat-row">
    <div class="stat"><div class="stat-num">18</div><div class="stat-lbl">Schemes</div></div>
    <div class="stat"><div class="stat-num">0.94</div><div class="stat-lbl">Task 1 score</div></div>
    <div class="stat"><div class="stat-num">0.91</div><div class="stat-lbl">Task 2 score</div></div>
    <div class="stat"><div class="stat-num">0.90</div><div class="stat-lbl">Task 3 score</div></div>
  </div>
</div>
<div class="main-grid">
  <div>
    <div class="card">
      <div class="card-title">Try a sample citizen profile</div>
      <div class="presets">
        <button class="preset-btn active" onclick="loadPreset('farmer',this)">Farmer - UP</button>
        <button class="preset-btn" onclick="loadPreset('student',this)">Student - Bihar</button>
        <button class="preset-btn" onclick="loadPreset('weaver',this)">Weaver - WB</button>
        <button class="preset-btn" onclick="loadPreset('senior',this)">Senior - MP</button>
        <button class="preset-btn" onclick="loadPreset('entrepreneur',this)">Entrepreneur - GJ</button>
      </div>
      <div class="card-title" style="margin-top:4px">Or enter details manually</div>
      <div class="row2">
        <div class="field"><label>Name</label><input id="f-name" value="Ramesh Kumar Yadav"></div>
        <div class="field"><label>Age</label><input id="f-age" type="number" value="38" min="1" max="100"></div>
      </div>
      <div class="row2">
        <div class="field"><label>Gender</label>
          <select id="f-gender"><option>Male</option><option>Female</option><option>Other</option></select></div>
        <div class="field"><label>Caste</label>
          <select id="f-caste"><option value="OBC">OBC</option><option value="SC">SC</option><option value="ST">ST</option><option value="General">General</option></select></div>
      </div>
      <div class="row2">
        <div class="field"><label>State</label><input id="f-state" value="Uttar Pradesh"></div>
        <div class="field"><label>Area</label>
          <select id="f-area"><option value="rural">Rural</option><option value="urban">Urban</option></select></div>
      </div>
      <div class="row2">
        <div class="field"><label>Occupation</label>
          <select id="f-occ">
            <option value="farmer">Farmer</option>
            <option value="student">Student</option>
            <option value="daily_wage_worker">Daily wage worker</option>
            <option value="weaver">Weaver</option>
            <option value="entrepreneur">Entrepreneur</option>
            <option value="retired">Retired</option>
          </select></div>
        <div class="field"><label>Annual income (Rs)</label><input id="f-income" type="number" value="85000"></div>
      </div>
      <div class="row2">
        <div class="field"><label>Land owned (acres)</label><input id="f-land" type="number" value="2.3" step="0.1" min="0"></div>
        <div class="field"><label>House type</label>
          <select id="f-house">
            <option value="kachha">Kachha</option>
            <option value="semi_pucca">Semi-pucca</option>
            <option value="pucca">Pucca</option>
            <option value="rented">Rented</option>
          </select></div>
      </div>
      <div class="field"><label>Status flags</label>
        <div class="checks-grid">
          <label class="check-item checked" id="lbl-aadhaar"><input type="checkbox" id="aadhaar" checked onchange="syncLabel('aadhaar')"> Has Aadhaar</label>
          <label class="check-item checked" id="lbl-bank"><input type="checkbox" id="bank" checked onchange="syncLabel('bank')"> Has bank account</label>
          <label class="check-item" id="lbl-lpg"><input type="checkbox" id="lpg" onchange="syncLabel('lpg')"> Has LPG connection</label>
          <label class="check-item" id="lbl-govt"><input type="checkbox" id="govt" onchange="syncLabel('govt')"> Govt employee</label>
          <label class="check-item" id="lbl-tax"><input type="checkbox" id="tax" onchange="syncLabel('tax')"> Income taxpayer</label>
          <label class="check-item" id="lbl-prof"><input type="checkbox" id="prof" onchange="syncLabel('prof')"> Professional</label>
        </div>
      </div>
      <button class="find-btn" id="findBtn" onclick="runCheck()">Find eligible schemes</button>
    </div>
  </div>
  <div>
    <div class="card">
      <div class="card-title">Eligible schemes</div>
      <div id="results">
        <div class="results-placeholder">
          <div class="dot-row"><div class="dot"></div><div class="dot"></div><div class="dot"></div></div>
          Select a sample profile or fill in the form
        </div>
      </div>
    </div>
  </div>
</div>
</div>
<script>
const SCHEMES=[
{id:"PM_KISAN",name:"PM-KISAN",ministry:"Ministry of Agriculture",benefit:6000,desc:"Rs 6,000/year in 3 installments for small and marginal farmers.",e:{occ:["farmer"],land_min:0.01,land_max:5,no_tax:true,no_govt:true,no_prof:true,needs_aadhaar:true,needs_bank:true,age_min:18}},
{id:"AYUSHMAN_BHARAT",name:"Ayushman Bharat PM-JAY",ministry:"Ministry of Health",benefit:500000,desc:"Rs 5 lakh/year health insurance cover per family.",e:{family_income_max:100000,needs_aadhaar:true,age_min:0}},
{id:"MGNREGA",name:"MGNREGA",ministry:"Ministry of Rural Development",benefit:57200,desc:"100 days of guaranteed wage employment per year.",e:{age_min:18,needs_aadhaar:true,needs_bank:true,rural_only:true}},
{id:"PM_AWAS_GRAMIN",name:"PM Awas Yojana - Gramin",ministry:"Ministry of Rural Development",benefit:120000,desc:"Rs 1.2 lakh housing subsidy for kachha house in rural areas.",e:{age_min:18,rural_only:true,needs_aadhaar:true,needs_bank:true,kachha_only:true}},
{id:"NSP_SC_SCHOLARSHIP",name:"SC Post-Matric Scholarship",ministry:"Ministry of Social Justice",benefit:15000,desc:"Rs 15,000/year for SC students in post-matric courses.",e:{occ:["student"],caste:["SC"],family_income_max:250000,age_max:30,needs_aadhaar:true,needs_bank:true}},
{id:"NSP_OBC_SCHOLARSHIP",name:"OBC Post-Matric Scholarship",ministry:"Ministry of Social Justice",benefit:12000,desc:"Rs 12,000/year for OBC students in post-matric courses.",e:{occ:["student"],caste:["OBC"],family_income_max:100000,age_max:30,needs_aadhaar:true,needs_bank:true}},
{id:"SUKANYA_SAMRIDDHI",name:"Sukanya Samriddhi Yojana",ministry:"Ministry of Finance",benefit:0,desc:"8.2% tax-free savings scheme for girl child (age 0-10).",e:{gender:["Female"],age_max:10,needs_aadhaar:true}},
{id:"UJJWALA_YOJANA",name:"PM Ujjwala Yojana",ministry:"Ministry of Petroleum",benefit:1600,desc:"Free LPG connection + subsidised cylinders for women below Rs 2 lakh income.",e:{gender:["Female"],family_income_max:200000,age_min:18,no_lpg:true,needs_aadhaar:true,needs_bank:true}},
{id:"KISAN_CREDIT_CARD",name:"Kisan Credit Card",ministry:"Ministry of Agriculture",benefit:0,desc:"Short-term credit at 4% interest for agricultural inputs.",e:{occ:["farmer"],land_min:0.01,age_min:18,age_max:75,needs_aadhaar:true,needs_bank:true}},
{id:"PMSBY",name:"PMSBY (Accident Insurance)",ministry:"Ministry of Finance",benefit:200000,desc:"Rs 2 lakh accidental death/disability cover at Rs 20/year.",e:{age_min:18,age_max:70,needs_bank:true}},
{id:"PMJJBY",name:"PMJJBY (Life Insurance)",ministry:"Ministry of Finance",benefit:200000,desc:"Rs 2 lakh life insurance cover at Rs 436/year.",e:{age_min:18,age_max:50,needs_bank:true}},
{id:"APY",name:"Atal Pension Yojana",ministry:"Ministry of Finance",benefit:60000,desc:"Guaranteed pension Rs 1,000-5,000/month after age 60.",e:{age_min:18,age_max:40,no_tax:true,no_govt:true,needs_aadhaar:true,needs_bank:true}},
{id:"PMEGP",name:"PMEGP",ministry:"Ministry of MSME",benefit:625000,desc:"15-35% subsidy on project cost for new enterprises.",e:{occ:["entrepreneur","daily_wage_worker"],age_min:18,needs_aadhaar:true,needs_bank:true}},
{id:"PMFBY",name:"PM Fasal Bima Yojana",ministry:"Ministry of Agriculture",benefit:0,desc:"Crop insurance at subsidised premium for farmers.",e:{occ:["farmer"],land_min:0.01,age_min:18,needs_aadhaar:true,needs_bank:true}},
{id:"STAND_UP_INDIA",name:"Stand-Up India",ministry:"Ministry of Finance",benefit:0,desc:"Bank loan Rs 10L-1Cr for SC/ST and women entrepreneurs.",e:{age_min:18,caste_or_female:true,needs_aadhaar:true,needs_bank:true}},
{id:"WEAVERS_MUDRA",name:"Weavers MUDRA Scheme",ministry:"Ministry of Textiles",benefit:0,desc:"Collateral-free loan up to Rs 10 lakh for handloom weavers.",e:{occ:["weaver"],age_min:18,needs_aadhaar:true,needs_bank:true}},
{id:"PMVVY",name:"PM Vaya Vandana Yojana",ministry:"Ministry of Finance",benefit:96000,desc:"Guaranteed 7.4% return pension scheme for senior citizens aged 60+.",e:{age_min:60,needs_aadhaar:true,needs_bank:true}}
];
const PRESETS={
farmer:{name:"Ramesh Kumar Yadav",age:38,gender:"Male",caste:"OBC",state:"Uttar Pradesh",area:"rural",occ:"farmer",income:85000,land:2.3,house:"kachha",aadhaar:true,bank:true,lpg:false,govt:false,tax:false,prof:false},
student:{name:"Priya Kumari",age:19,gender:"Female",caste:"SC",state:"Bihar",area:"urban",occ:"student",income:0,land:0,house:"rented",aadhaar:true,bank:true,lpg:true,govt:false,tax:false,prof:false},
weaver:{name:"Fatima Begum",age:35,gender:"Female",caste:"OBC",state:"West Bengal",area:"rural",occ:"weaver",income:72000,land:0.2,house:"kachha",aadhaar:true,bank:true,lpg:false,govt:false,tax:false,prof:false},
senior:{name:"Saraswati Bai Patel",age:63,gender:"Female",caste:"General",state:"Madhya Pradesh",area:"urban",occ:"retired",income:0,land:0,house:"pucca",aadhaar:true,bank:true,lpg:true,govt:false,tax:false,prof:false},
entrepreneur:{name:"Rahul Mehta",age:27,gender:"Male",caste:"SC",state:"Gujarat",area:"urban",occ:"entrepreneur",income:150000,land:0,house:"rented",aadhaar:true,bank:true,lpg:true,govt:false,tax:false,prof:false}
};
function loadPreset(key,btn){
  document.querySelectorAll('.preset-btn').forEach(b=>b.classList.remove('active'));
  if(btn)btn.classList.add('active');
  const p=PRESETS[key];
  document.getElementById('f-name').value=p.name;
  document.getElementById('f-age').value=p.age;
  document.getElementById('f-gender').value=p.gender;
  document.getElementById('f-caste').value=p.caste;
  document.getElementById('f-state').value=p.state;
  document.getElementById('f-area').value=p.area;
  document.getElementById('f-occ').value=p.occ;
  document.getElementById('f-income').value=p.income;
  document.getElementById('f-land').value=p.land;
  document.getElementById('f-house').value=p.house;
  ['aadhaar','bank','lpg','govt','tax','prof'].forEach(id=>{
    document.getElementById(id).checked=p[id];
    const lbl=document.getElementById('lbl-'+id);
    if(lbl)lbl.classList.toggle('checked',p[id]);
  });
  runCheck();
}
function syncLabel(id){
  const cb=document.getElementById(id);
  const lbl=document.getElementById('lbl-'+id);
  if(lbl)lbl.classList.toggle('checked',cb.checked);
}
function getProfile(){
  return{age:parseInt(document.getElementById('f-age').value)||0,gender:document.getElementById('f-gender').value,caste:document.getElementById('f-caste').value,area:document.getElementById('f-area').value,occ:document.getElementById('f-occ').value,income:parseInt(document.getElementById('f-income').value)||0,land:parseFloat(document.getElementById('f-land').value)||0,house:document.getElementById('f-house').value,aadhaar:document.getElementById('aadhaar').checked,bank:document.getElementById('bank').checked,lpg:document.getElementById('lpg').checked,govt:document.getElementById('govt').checked,tax:document.getElementById('tax').checked,prof:document.getElementById('prof').checked};
}
function isEligible(p,e){
  if(e.age_min!==undefined&&p.age<e.age_min)return false;
  if(e.age_max!==undefined&&p.age>e.age_max)return false;
  if(e.gender&&!e.gender.includes(p.gender))return false;
  if(e.caste&&!e.caste.includes(p.caste))return false;
  if(e.occ&&!e.occ.includes(p.occ))return false;
  if(e.rural_only&&p.area!=='rural')return false;
  if(e.kachha_only&&p.house!=='kachha')return false;
  if(e.no_lpg&&p.lpg)return false;
  if(e.no_tax&&p.tax)return false;
  if(e.no_govt&&p.govt)return false;
  if(e.no_prof&&p.prof)return false;
  if(e.needs_aadhaar&&!p.aadhaar)return false;
  if(e.needs_bank&&!p.bank)return false;
  if(e.land_min!==undefined&&p.land<e.land_min)return false;
  if(e.land_max!==undefined&&p.land>e.land_max)return false;
  if(e.family_income_max!==undefined&&p.income>e.family_income_max)return false;
  if(e.caste_or_female){if(!['SC','ST'].includes(p.caste)&&p.gender!=='Female')return false;}
  return true;
}
function runCheck(){
  const p=getProfile();
  const btn=document.getElementById('findBtn');
  btn.disabled=true;btn.textContent='Checking...';
  document.getElementById('results').innerHTML='<div class="spinner"><div class="spin-dot"></div><div class="spin-dot"></div><div class="spin-dot"></div></div>';
  setTimeout(()=>{
    const eligible=SCHEMES.filter(s=>isEligible(p,s.e));
    const monetary=eligible.filter(s=>s.benefit>0).sort((a,b)=>b.benefit-a.benefit);
    const access=eligible.filter(s=>s.benefit===0).sort((a,b)=>a.name.localeCompare(b.name));
    const ranked=[...monetary,...access];
    const maxBenefit=monetary.length?monetary[0].benefit:1;
    let html='';
    if(ranked.length===0){html='<div style="text-align:center;padding:32px 0;color:#aaa;font-size:13px">No eligible schemes found. Try adjusting the profile.</div>';}
    else{
      const total=ranked.reduce((s,x)=>s+x.benefit,0);
      html+='<div class="result-header"><div class="result-count">'+ranked.length+' schemes found</div><div class="result-sub">Total potential benefit: Rs '+total.toLocaleString('en-IN')+'/year</div><div class="pill-row">';
      if(monetary.length)html+='<span class="pill pg">'+monetary.length+' monetary</span>';
      if(access.length)html+='<span class="pill pb">'+access.length+' access/credit</span>';
      html+='</div></div>';
      ranked.forEach((s,i)=>{
        const isTop=i===0;
        const barPct=s.benefit>0?Math.round((s.benefit/maxBenefit)*100):15;
        const barColor=isTop?'#1D9E75':i<3?'#5DCAA5':'#9FE1CB';
        const benefitText=s.benefit>0?'Rs '+s.benefit.toLocaleString('en-IN')+'/yr':'Access benefit';
        const pillCls=s.benefit>0?'benefit-pill':'benefit-pill access';
        const rnCls=isTop?'rn gold':'rn';
        html+='<div class="scheme-card'+(isTop?' rank1':'')+'"><div class="sc-top"><div class="sc-name"><span class="'+rnCls+'">'+(i+1)+'</span>'+s.name+'</div><span class="'+pillCls+'">'+benefitText+'</span></div><div class="sc-ministry">'+s.ministry+'</div><div class="sc-desc">'+s.desc+'</div>';
        if(s.benefit>0){html+='<div class="sc-bar"><div class="bar-track"><div class="bar-fill" style="width:'+barPct+'%;background:'+barColor+'"></div></div><span class="bar-label">'+barPct+'%</span></div>';}
        html+='</div>';
      });
    }
    document.getElementById('results').innerHTML=html;
    btn.disabled=false;btn.textContent='Find eligible schemes';
  },600);
}
window.onload=()=>loadPreset('farmer',document.querySelector('.preset-btn.active'));
</script>
</body>
</html>"""


@app.get("/health")
def health():
    return {"status": "ok", "env": "govscheme-env", "version": "1.0.0"}


@app.post("/reset")
def reset(req: ResetRequest = ResetRequest()):
    valid = ["scheme_identification", "scheme_ranking", "form_filling"]
    if req.task_name not in valid:
        raise HTTPException(400, f"Invalid task_name. Choose from: {valid}")
    obs = _env.reset(
        task_name=req.task_name,
        citizen_id=req.citizen_id,
        seed=req.seed,
    )
    return JSONResponse({
        "observation": _obs_to_dict(obs),
        "reward": 0.0,
        "done": False,
        "info": {"citizen_id": _env.state.citizen_id},
    })


@app.post("/step")
def step(req: StepRequest):
    action = GovSchemeAction(
        action_type=req.action_type,
        scheme_ids=req.scheme_ids,
        ranked_schemes=req.ranked_schemes,
        form_data=req.form_data,
        reasoning=req.reasoning,
    )
    obs, reward, done, info = _env.step(action)
    return JSONResponse({
        "observation": _obs_to_dict(obs),
        "reward": reward,
        "done": done,
        "info": info,
    })


@app.get("/state")
def state():
    return JSONResponse(_state_to_dict(_env.state))


@app.get("/tasks")
def tasks():
    return {"tasks": GovSchemeEnvironment.list_tasks()}


@app.get("/schemes")
def schemes():
    import json
    from pathlib import Path
    data = json.loads((Path(__file__).parent / "schemes.json").read_text())
    return {
        "count": len(data),
        "schemes": [
            {
                "scheme_id": s["scheme_id"],
                "name": s["name"],
                "benefit_type": s["benefit_type"],
                "annual_benefit_inr": s["annual_benefit_inr"],
                "benefit_description": s["benefit_description"],
            }
            for s in data
        ],
    }


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 7860))
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860)

if __name__ == "__main__":
    main()