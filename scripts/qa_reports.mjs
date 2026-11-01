// Offline generated-report QA with Node built-ins and an installed Chromium.
import {spawn} from 'node:child_process';
import {mkdtemp,readFile,writeFile,mkdir,rm} from 'node:fs/promises';
import {tmpdir} from 'node:os';
import {join,dirname,resolve} from 'node:path';
import {pathToFileURL,fileURLToPath} from 'node:url';
import {createHash} from 'node:crypto';

const root=resolve(dirname(fileURLToPath(import.meta.url)),'..');
const outputIndex=process.argv.indexOf('--output');
const output=outputIndex>=0?resolve(process.argv[outputIndex+1]):join(tmpdir(),'neuroedge-report-qa');
if(!process.env.CHROMIUM_BIN)throw Error('Set CHROMIUM_BIN to the installed Chromium executable.');
await mkdir(output,{recursive:true});
const profile=await mkdtemp(join(tmpdir(),'neuroedge-report-profile-'));
const chromium=spawn(process.env.CHROMIUM_BIN,[
  '--headless','--no-sandbox','--disable-dev-shm-usage','--disable-gpu',
  '--no-zygote','--hide-scrollbars','--allow-file-access-from-files',
  '--remote-debugging-port=0','--remote-debugging-address=127.0.0.1',
  '--user-data-dir='+profile,'about:blank'],{stdio:['ignore','ignore','pipe']});
let socket;
const reports=['covariance_lab_corrected','session_adaptation','research_summary','context_fusion','gated_context','personalization','personalization_gated','residual_personal','frontier'];
const evidence={scope:'Offline generated reports only; no EEG fitting or external browsing.',
  qa_source_sha256:createHash('sha256').update(await readFile(fileURLToPath(import.meta.url))).digest('hex'),checks:[]};
try {
  const endpoint=await new Promise((resolve,reject)=>{
    let buffer='';const timer=setTimeout(()=>reject(Error('Chromium start timed out: '+buffer.slice(-1000))),15000);
    chromium.stderr.on('data',chunk=>{buffer+=chunk;const match=buffer.match(/DevTools listening on (ws:\/\/\S+)/);if(match){clearTimeout(timer);resolve(match[1]);}});
    chromium.once('exit',code=>{clearTimeout(timer);reject(Error('Chromium exited '+code+': '+buffer.slice(-1000)));});
  });
  socket=new WebSocket(endpoint);
  await new Promise((resolve,reject)=>{socket.addEventListener('open',resolve,{once:true});socket.addEventListener('error',reject,{once:true});});
  let id=0;const pending=new Map();
  socket.addEventListener('message',event=>{const value=JSON.parse(event.data);if(value.id&&pending.has(value.id)){const item=pending.get(value.id);pending.delete(value.id);clearTimeout(item.timer);value.error?item.reject(Error(JSON.stringify(value.error))):item.resolve(value.result);}});
  const call=(method,params={},sessionId)=>new Promise((resolve,reject)=>{
    const current=++id;const timer=setTimeout(()=>{pending.delete(current);reject(Error('CDP timed out: '+method));},20000);
    pending.set(current,{resolve,reject,timer});socket.send(JSON.stringify({id:current,method,params,...(sessionId?{sessionId}:{})}));
  });
  evidence.browser=await call('Browser.getVersion');
  const {targetId}=await call('Target.createTarget',{url:'about:blank'});
  const {sessionId}=await call('Target.attachToTarget',{targetId,flatten:true});
  const page=(method,params)=>call(method,params,sessionId);
  await page('Page.enable');
  for(const width of [1280,390]) {
    await page('Emulation.setDeviceMetricsOverride',{width,height:900,deviceScaleFactor:1,mobile:width<500});
    for(const report of reports) {
      await readFile(join(root,'runs',report,'results.html'));
      const url=pathToFileURL(join(root,'runs',report,'results.html')).href;
      await page('Page.navigate',{url});
      let ready=false;
      for(let tries=0;tries<60;tries++) {
        const result=await page('Runtime.evaluate',{expression:'location.href==='+JSON.stringify(url)+' && document.readyState==="complete" && [...document.images].every(x=>x.complete)',returnByValue:true});
        if(result.result.value){ready=true;break;}
        await new Promise(resolve=>setTimeout(resolve,100));
      }
      if(!ready)throw Error('Page/images did not become ready: '+report);
      const {result}=await page('Runtime.evaluate',{expression:`JSON.stringify({title:document.title,language:document.documentElement.lang,direction:document.documentElement.dir||'ltr',viewport:innerWidth,documentWidth:document.documentElement.scrollWidth,bodyWidth:document.body.scrollWidth,images:[...document.images].map(x=>({loaded:x.naturalWidth>0,width:x.naturalWidth,height:x.naturalHeight})),tables:document.querySelectorAll('table').length,textLength:document.body.innerText.length})`,returnByValue:true});
      const stats=JSON.parse(result.value);
      if(stats.language!=='en')throw Error('Report language is not English: '+report);
      if(report==='session_adaptation') {
        const filter=await page('Runtime.evaluate',{expression:`(()=>{const s=document.getElementById('budget');s.value='16';s.dispatchEvent(new Event('change'));const rows=[...document.querySelectorAll('#detail tbody tr')].filter(r=>r.style.display!=='none');s.value='all';s.dispatchEvent(new Event('change'));return rows.length})()`,returnByValue:true});
        if(filter.result.value!==11)throw Error('Budget filter does not show exactly 11 methods');
      }
      const layout=await page('Page.getLayoutMetrics');
      const height=Math.ceil(layout.cssContentSize.height);
      const {data}=await page('Page.captureScreenshot',{format:'png',captureBeyondViewport:true,clip:{x:0,y:0,width,height:Math.min(height,10000),scale:1}});
      await writeFile(join(output,report+'_'+width+'.png'),Buffer.from(data,'base64'));
      const entry={report,width,...stats,height,no_page_horizontal_overflow:stats.documentWidth<=width+1,all_images_loaded:stats.images.length>0&&stats.images.every(x=>x.loaded)};
      evidence.checks.push(entry);console.log(JSON.stringify(entry));
      if(!entry.no_page_horizontal_overflow||!entry.all_images_loaded||stats.tables<1||stats.textLength<100)throw Error('Report failed layout/content QA: '+report+' '+width);
    }
  }
  evidence.status='PASS';evidence.checked_utc=new Date().toISOString();
  await writeFile(join(root,'tests/final_report_browser_qa.json'),JSON.stringify(evidence,null,2));
  await call('Browser.close');
} finally {
  socket?.close();
  if(chromium.exitCode===null)chromium.kill('SIGTERM');
  await new Promise(resolve=>setTimeout(resolve,250));
  await rm(profile,{recursive:true,force:true}).catch(()=>{});
}
