import { useEffect, useState } from 'react'

const DAYS = ['Monday','Tuesday','Wednesday','Thursday','Friday','Saturday','Sunday']

export default function Home() {
  const [apiUrl] = useState(process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
  const [graph, setGraph] = useState(null)
  const [loadingGraph, setLoadingGraph] = useState(false)
  const [error, setError] = useState(null)

  // form state
  const [source, setSource] = useState('B')
  const [dest, setDest] = useState('N')
  const [hour, setHour] = useState(14)
  const [day, setDay] = useState(3)
  const [weather, setWeather] = useState('clear')
  const [traffic, setTraffic] = useState(0.5)
  const [threshold, setThreshold] = useState(0.7)

  const [planning, setPlanning] = useState(false)
  const [result, setResult] = useState(null)

  useEffect(() => {
    async function loadGraph(){
      setLoadingGraph(true)
      try{
        const res = await fetch(`${apiUrl}/graph`)
        if(!res.ok) throw new Error('Failed to fetch graph')
        const j = await res.json()
        setGraph(j)
        // set default nodes
        const nodes = Object.keys(j.nodes)
        if(nodes.length>=2){
          setSource(nodes[1]||nodes[0])
          setDest(nodes[nodes.length-1])
        }
      }catch(e){
        setError(e.message)
      }finally{setLoadingGraph(false)}
    }
    loadGraph()
  },[apiUrl])

  async function handlePlan(e){
    e.preventDefault()
    setPlanning(true)
    setResult(null)
    setError(null)
    try{
      const body = { source, dest, hour: Number(hour), day: Number(day), weather, traffic: Number(traffic), risk_threshold: Number(threshold) }
      const res = await fetch(`${apiUrl}/route`, { method: 'POST', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) })
      if(!res.ok){
        const txt = await res.text()
        throw new Error(`API error: ${res.status} ${txt}`)
      }
      const j = await res.json()
      setResult(j)
    }catch(e){
      setError(e.message)
    }finally{setPlanning(false)}
  }

  // helpers to draw SVG map
  function getBounds(nodes){
    const xs = Object.values(nodes).map(n=>n[0])
    const ys = Object.values(nodes).map(n=>n[1])
    const minX = Math.min(...xs), maxX = Math.max(...xs)
    const minY = Math.min(...ys), maxY = Math.max(...ys)
    return {minX, maxX, minY, maxY}
  }

  function coordToSvg(x,y,bounds,w=800,h=600,pad=40){
    const {minX,maxX,minY,maxY} = bounds
    const sx = (w-2*pad)/(maxX-minX || 1)
    const sy = (h-2*pad)/(maxY-minY || 1)
    const scale = Math.min(sx,sy)
    const tx = pad - minX*scale
    const ty = pad - minY*scale
    return [x*scale+tx, h - (y*scale+ty)]
  }

  function edgeColor(r){
    if(r===null || r===undefined) return '#777'
    if(r>0.7) return '#e74c3c'
    if(r>0.4) return '#f39c12'
    return '#2ecc71'
  }

  return (
    <div className="page">
      <aside className="sidebar">
        <h1>Smart Traffic Route Planner</h1>
        <form onSubmit={handlePlan} className="form">
          <label>Source</label>
          <select value={source} onChange={e=>setSource(e.target.value)}>
            {graph && Object.keys(graph.nodes).map(n=> <option key={n} value={n}>{n}</option>)}
          </select>

          <label>Destination</label>
          <select value={dest} onChange={e=>setDest(e.target.value)}>
            {graph && Object.keys(graph.nodes).filter(n=>n!==source).map(n=> <option key={n} value={n}>{n}</option>)}
          </select>

          <label>Hour: {hour}</label>
          <input type="range" min="0" max="23" value={hour} onChange={e=>setHour(e.target.value)} />

          <label>Day</label>
          <select value={day} onChange={e=>setDay(e.target.value)}>
            {DAYS.map((d,i)=> <option key={d} value={i}>{d}</option>)}
          </select>

          <label>Weather</label>
          <select value={weather} onChange={e=>setWeather(e.target.value)}>
            <option value="clear">Clear</option>
            <option value="rain">Rain</option>
            <option value="fog">Fog</option>
          </select>

          <label>Traffic: {Math.round(traffic*100)}%</label>
          <input type="range" min="0.1" max="1.0" step="0.05" value={traffic} onChange={e=>setTraffic(e.target.value)} />

          <label>Risk Threshold: {threshold}</label>
          <input type="range" min="0.5" max="0.9" step="0.05" value={threshold} onChange={e=>setThreshold(e.target.value)} />

          <button type="submit" disabled={planning || loadingGraph}>{planning? 'Planning...': 'Find Safe Route'}</button>
        </form>

        <div className="info">
          <p>Backend: <code>{apiUrl}</code></p>
          {error && <p className="error">{error}</p>}
          {result && (
            <div className="result">
              <h3>Result: {result.action}</h3>
              <p>{result.message}</p>
              <p>Cost: {result.cost.toFixed(2)} | Shortest: {result.shortest_cost?.toFixed(2)}</p>
              <h4>Path</h4>
              <pre>{result.path.join(' → ')}</pre>
              {result.avoided_edges && result.avoided_edges.length>0 && (
                <div>
                  <h4>Avoided Edges</h4>
                  <ul>{result.avoided_edges.map(([a,b])=> <li key={`${a}-${b}`}>{a} → {b}</li>)}</ul>
                </div>
              )}
            </div>
          )}
        </div>
      </aside>

      <main className="main">
        <h2>Map</h2>
        <div className="map">
          {graph ? (()=>{
            const nodes = graph.nodes
            const edges = graph.edges
            const bounds = getBounds(nodes)
            // build risk map from result
            const riskMap = {}
            if(result && result.edge_risks){
              result.edge_risks.forEach(er=>{ riskMap[`${er.edge[0]}-${er.edge[1]}`]=er.risk })
            }
            return (
              <svg width={900} height={700} style={{background:'#0f0f1a',borderRadius:8}}>
                {/* edges */}
                {edges.map((e,i)=>{
                  const u = e.u, v = e.v
                  const nu = nodes[u], nv = nodes[v]
                  const [x1,y1] = coordToSvg(nu[0], nu[1], bounds, 900,700)
                  const [x2,y2] = coordToSvg(nv[0], nv[1], bounds, 900,700)
                  const r = riskMap[`${u}-${v}`] ?? riskMap[`${v}-${u}`]
                  const color = edgeColor(r)
                  return <line key={i} x1={x1} y1={y1} x2={x2} y2={y2} stroke={color} strokeWidth={2} opacity={0.9} />
                })}

                {/* shortest path overlay (dashed blue) */}
                {result && result.shortest_path && (
                  result.shortest_path.map((n,idx)=>{
                    if(idx===result.shortest_path.length-1) return null
                    const a = result.shortest_path[idx], b = result.shortest_path[idx+1]
                    const [x1,y1] = coordToSvg(graph.nodes[a][0], graph.nodes[a][1], bounds,900,700)
                    const [x2,y2] = coordToSvg(graph.nodes[b][0], graph.nodes[b][1], bounds,900,700)
                    return <line key={`sp-${idx}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#3498db" strokeWidth={4} strokeDasharray="6,4" opacity={0.6} />
                  })
                )}

                {/* chosen path overlay (green) */}
                {result && result.path && (
                  result.path.map((n,idx)=>{
                    if(idx===result.path.length-1) return null
                    const a = result.path[idx], b = result.path[idx+1]
                    const [x1,y1] = coordToSvg(graph.nodes[a][0], graph.nodes[a][1], bounds,900,700)
                    const [x2,y2] = coordToSvg(graph.nodes[b][0], graph.nodes[b][1], bounds,900,700)
                    return <line key={`cp-${idx}`} x1={x1} y1={y1} x2={x2} y2={y2} stroke="#00ff88" strokeWidth={6} opacity={0.95} />
                  })
                )}

                {/* nodes */}
                {Object.entries(graph.nodes).map(([k,pos])=>{
                  const [x,y] = coordToSvg(pos[0], pos[1], getBounds(graph.nodes),900,700)
                  const isSource = result && result.path && result.path[0]===k
                  const isDest = result && result.path && result.path[result.path.length-1]===k
                  return (
                    <g key={k}>
                      <circle cx={x} cy={y} r={isSource||isDest?10:6} fill={isSource? '#00ff88' : (isDest? '#ff6b6b' : '#e0e0e0')} />
                      <text x={x} y={y+4} textAnchor="middle" fontSize={isSource||isDest?12:10} fill={isSource||isDest? '#0f0f1a' : '#0f0f1a'} fontWeight="bold">{k}</text>
                    </g>
                  )
                })}
              </svg>
            )
          })() : (
            <div className="loading">{loadingGraph? 'Loading map...':'No graph available'}</div>
          )}
        </div>
      </main>

      <style jsx>{`
        .page{display:flex;min-height:100vh;color:#e0e0e0;background:#0f0f1a}
        .sidebar{width:360px;padding:24px;background:#16213e;border-right:1px solid #222}
        .sidebar h1{color:#00ff88}
        .form{display:flex;flex-direction:column;gap:8px}
        label{font-size:14px;margin-top:6px}
        select,input[type=range]{width:100%}
        button{margin-top:12px;padding:10px;background:#00ff88;border:none;border-radius:6px;color:#081018;font-weight:bold}
        .info{margin-top:18px;font-size:13px}
        .main{flex:1;padding:18px}
        .map{background:#0f0f1a;padding:8px;border-radius:8px}
        pre{background:#111;color:#e0e0e0;padding:8px;border-radius:6px}
        .error{color:#f39c12}
      `}</style>
    </div>
  )
}
