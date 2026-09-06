import React,{useMemo,useState} from 'react';
import {createRoot} from 'react-dom/client';
import Plot from 'react-plotly.js';
import './style.css';

const depths=[0,5,10,20,30,50,75,100,125,150,200,300,500,700,1000];
const lat=Array.from({length:101},(_,i)=>+(5+i*0.25).toFixed(2));
const lon=Array.from({length:241},(_,i)=>+(45+i*0.25).toFixed(2));

function App(){
 const [depth,setDepth]=useState(100);
 const [point,setPoint]=useState({lat:15,lon:85});
 const fake=useMemo(()=>depths.map(d=>30-0.018*d+0.7*Math.sin(d/120)),[]);
 return <div className="app">
  <header><div><div className="eyebrow">SIH 2026 • INCOIS</div><h1>OceanEmbed AI</h1><p>Satellite surface observations → latent ocean embedding → 15-level subsurface temperature.</p></div><div className="badge">NORTH INDIAN OCEAN</div></header>
  <section className="grid">
   <aside className="panel"><h2>Reconstruction</h2><label>Region</label><select defaultValue="nio"><option value="nio">North Indian Ocean</option><option>Bay of Bengal</option><option>Arabian Sea</option></select>
   <label>Date</label><input type="date" defaultValue="2020-08-15"/>
   <label>Depth: {depth} m</label><input type="range" min="0" max="1000" step="5" value={depth} onChange={e=>setDepth(+e.target.value)}/>
   <div className="depths">{depths.map(d=><button className={d===depth?'active':''} key={d} onClick={()=>setDepth(d)}>{d}</button>)}</div>
   <button className="primary">RECONSTRUCT</button>
   <div className="mini"><b>Grid</b><span>101 × 241</span></div><div className="mini"><b>Input channels</b><span>7</span></div><div className="mini"><b>Output depths</b><span>15</span></div>
   </aside>
   <main className="panel"><div className="panelhead"><h2>Predicted Temperature Map</h2><span>{depth} m</span></div><div className="map"><div className="gridmap">{Array.from({length:120},(_,i)=><div key={i} style={{opacity:0.45+0.55*((i%17)/16)}}/> )}</div><div className="maplabel top">30°N</div><div className="maplabel bottom">5°N</div><div className="maplabel left">45°E</div><div className="maplabel right">105°E</div><div className="point" title="click location"/></div></main>
  </section>
  <section className="grid lower"><div className="panel"><h2>Vertical Temperature Profile</h2><Plot data={[{x:fake,y:depths,mode:'lines+markers',line:{shape:'spline'}}]} layout={{paper_bgcolor:'transparent',plot_bgcolor:'transparent',margin:{l:55,r:15,t:10,b:45},xaxis:{title:'Temperature (°C)'},yaxis:{title:'Depth (m)',autorange:'reversed',gridcolor:'rgba(255,255,255,.12)'}}} style={{width:'100%',height:420}} config={{displayModeBar:false}}/></div><div className="panel"><h2>Evaluation</h2><div className="metrics"><div><b>RMSE</b><strong>—</strong></div><div><b>MAE</b><strong>—</strong></div><div><b>Bias</b><strong>—</strong></div><div><b>Correlation</b><strong>—</strong></div></div><p className="muted">Live values will populate after the trained model is evaluated against the held-out period and independent ARGO profiles.</p></div></section>
  <footer>OceanEmbed • Daily 0.25° • 5°N–30°N • 45°E–105°E • 15 standard depths through 1000 m</footer>
 </div>
}
createRoot(document.getElementById('root')).render(<App/>);
