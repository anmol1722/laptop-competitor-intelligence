/* Data-only charts for assistant replies. No provider HTML or executable chart code. */
(function () {
  'use strict';
  var NS = 'http://www.w3.org/2000/svg';
  var palette = ['#7763cd', '#4e83b1', '#bf7758', '#a367ae', '#568d91', '#998149', '#7676a5', '#9e6478', '#5c7994', '#868970', '#8a849b'];
  var serial = 0;
  function esc(value) { return String(value == null ? '' : value).replace(/[&<>"']/g, function (c) { return ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'})[c]; }); }
  function num(value, digits) { return Number(value).toLocaleString('en-US', {maximumFractionDigits: digits == null ? 1 : digits}); }
  function metric(value, unit) { return (unit === 'USD' ? '$' : '') + num(value, unit === 'USD' ? 2 : 1) + (unit === '%' ? '%' : ''); }
  function cut(value, length) { value=String(value); return value.length > length ? value.slice(0,length-1)+'…' : value; }
  function valid(chart) {
    return chart && ['bar','scatter','histogram','donut'].indexOf(chart.type)>=0 &&
      Array.isArray(chart.rows) && chart.rows.length>0 && chart.rows.length<=200 &&
      chart.rows.every(function (r) { return typeof r.label==='string' && typeof r.value==='number' && Number.isFinite(r.value) && r.value>=0 && (chart.type!=='scatter' || (typeof r.x==='number' && Number.isFinite(r.x) && r.x>0)); });
  }
  function description(chart, row) {
    if (chart.type==='scatter') return row.label+': $'+num(row.x,2)+', '+num(row.value,2)+' / 5, '+num(row.n,0)+' Amazon ratings';
    var text=row.label+': '+metric(row.value,chart.unit);
    if (chart.unit==='USD') text+='; '+num(row.n,0)+' priced / '+num(row.total,0)+' matching listings ('+num(row.coverage*100,1)+'% coverage)';
    else if (chart.unit==='%') text+='; '+num(row.n,0)+' '+(chart.sample_label || 'analysed review mentions');
    if (row.reliable===false) text+='; small sample';
    return text;
  }
  function point(chart,row,index,body) {
    return '<g class="chat-chart__point" data-point="'+index+'" tabindex="0" role="button" aria-label="'+esc(description(chart,row))+'"><title>'+esc(description(chart,row))+'</title>'+body+'</g>';
  }
  function svg(chart,width,id) {
    width=Math.max(260,Math.round(width));
    var rows=chart.rows,height=0,content='',ink='var(--ink)',muted='var(--ink-muted)',grid='var(--grid)';
    if (chart.type==='bar') {
      var left=Math.min(200,Math.max(82,width*.30)),right=chart.unit==='USD'?74:55,space=width-left-right;
      var max=chart.unit==='%'?100:Math.max.apply(null,rows.map(function (r){return r.value;}))*1.03 || 1;
      height=rows.length*43+22;
      rows.forEach(function (r,i) {
        var y=i*43+10,w=Math.max(0,space*r.value/max);
        content+=point(chart,r,i,'<rect x="0" y="'+(y-6)+'" width="'+width+'" height="42" fill="transparent"/>'+
          '<text x="0" y="'+(y+12)+'" font-size="11" fill="'+ink+'">'+esc(cut(r.label,Math.floor(left/6)))+'</text>'+
          '<rect x="'+left+'" y="'+y+'" width="'+space+'" height="17" rx="3" fill="var(--surface-3)"/>'+
          '<rect x="'+left+'" y="'+y+'" width="'+w+'" height="17" rx="3" fill="'+palette[0]+'" opacity="'+(r.reliable===false?'.48':'1')+'"/>'+
          '<text x="'+(width-1)+'" y="'+(y+13)+'" text-anchor="end" font-size="11" fill="'+ink+'">'+esc(metric(r.value,chart.unit))+'</text>'+
          '<text x="'+left+'" y="'+(y+32)+'" font-size="9" fill="'+muted+'">'+esc(chart.unit==='USD'?num(r.n,0)+' priced · '+num(r.coverage*100,0)+'% coverage'+(r.reliable===false?' · small sample':'') : chart.unit==='%'?num(r.n,0)+' review mentions':num(r.n,0)+' '+chart.unit)+'</text>');
      });
    } else if (chart.type==='scatter') {
      height=260;
      var x0=39,xspan=width-57,y0=15,yspan=203,xmax=Math.max.apply(null,rows.map(function(r){return r.x;}))*1.05 || 1;
      for(var t=1;t<=5;t++) { var y=y0+yspan-(t-1)/4*yspan; content+='<path d="M'+x0+' '+y+'H'+(width-10)+'" stroke="'+grid+'"/><text x="26" y="'+(y+4)+'" text-anchor="end" font-size="10" fill="'+muted+'">'+t+'</text>'; }
      for(var j=0;j<=3;j++){var v=xmax*j/3,x=x0+xspan*j/3;content+='<text x="'+x+'" y="239" text-anchor="'+(j===3?'end':'middle')+'" font-size="10" fill="'+muted+'">$'+num(v,0)+'</text>';}
      rows.forEach(function(r,i){content+=point(chart,r,i,'<circle cx="'+(x0+r.x/xmax*xspan)+'" cy="'+(y0+yspan-(r.value-1)/4*yspan)+'" r="5" fill="'+palette[0]+'" fill-opacity=".7" stroke="var(--surface-1)" stroke-width="1"/>');});
    } else if (chart.type==='histogram') {
      height=250;
      var maximum=Math.max.apply(null,rows.map(function(r){return r.value;})) || 1;
      var start=37,span=width-47,step=span/rows.length;
      for(var k=0;k<=3;k++){var yy=205-k*185/3;content+='<path d="M'+start+' '+yy+'H'+(width-8)+'" stroke="'+grid+'"/><text x="30" y="'+(yy+4)+'" text-anchor="end" font-size="9" fill="'+muted+'">'+num(maximum*k/3,0)+'</text>';}
      rows.forEach(function(r,i){var hh=r.value/maximum*185;content+=point(chart,r,i,'<rect x="'+(start+i*step+1)+'" y="'+(205-hh)+'" width="'+Math.max(1,step-2)+'" height="'+Math.max(1,hh)+'" fill="'+palette[0]+'"/>');});
      content+='<text x="'+start+'" y="228" font-size="10" fill="'+muted+'">$'+num(rows[0].low,0)+'</text><text x="'+(width-8)+'" y="228" text-anchor="end" font-size="10" fill="'+muted+'">$'+num(rows[rows.length-1].high,0)+'</text>';
    } else {
      height=202+rows.length*23;
      var total=rows.reduce(function(s,r){return s+r.value;},0),offset=0,circ=2*Math.PI*65;
      rows.forEach(function(r,i){var part=r.value/total*circ;content+=point(chart,r,i,'<circle cx="'+(width/2)+'" cy="91" r="65" fill="none" stroke="'+palette[i%palette.length]+'" stroke-width="27" stroke-dasharray="'+part+' '+(circ-part)+'" stroke-dashoffset="'+(-offset)+'" transform="rotate(-90 '+(width/2)+' 91)"/>');offset+=part;});
      content+='<text x="'+(width/2)+'" y="88" text-anchor="middle" fill="'+ink+'" font-size="21" font-weight="600">'+num(total,0)+'</text><text x="'+(width/2)+'" y="107" text-anchor="middle" fill="'+muted+'" font-size="10">'+esc(chart.unit)+'</text>';
      rows.forEach(function(r,i){var y=204+i*23;content+=point(chart,r,i,'<rect x="0" y="'+(y-10)+'" width="10" height="10" rx="2" fill="'+palette[i%palette.length]+'"/><text x="18" y="'+y+'" fill="'+ink+'" font-size="11">'+esc(cut(r.label,Math.floor((width-110)/6)))+'</text><text x="'+(width-2)+'" y="'+y+'" text-anchor="end" fill="'+muted+'" font-size="10">'+num(r.value,0)+' · '+num(r.value/total*100,1)+'%</text>');});
    }
    return '<svg xmlns="'+NS+'" viewBox="0 0 '+width+' '+height+'" width="'+width+'" height="'+height+'" role="img" aria-labelledby="'+id+'-title '+id+'-desc" style="font-family:system-ui,sans-serif;max-width:100%;height:auto"><title id="'+id+'-title">'+esc(chart.title)+'</title><desc id="'+id+'-desc">'+esc(chart.scope)+'. Exact values are available in the data table.</desc><rect width="100%" height="100%" fill="var(--surface-1)"/>'+content+'</svg>';
  }
  function table(chart) {
    return '<details class="chat-chart__data"><summary>View exact data ('+chart.rows.length+' rows)</summary><div class="chat-chart__table-wrap"><table><thead><tr><th>Group / product</th>'+(chart.type==='scatter'?'<th>Price (USD)</th>':'')+'<th>'+esc(chart.unit==='%'?(chart.polarity==='negative'?'Negative':'Positive')+' share (%)':chart.type==='scatter'?'Rating / 5':chart.unit)+'</th><th>Sample</th></tr></thead><tbody>'+chart.rows.map(function(r){return '<tr><th scope="row">'+esc(r.label)+'</th>'+(chart.type==='scatter'?'<td>'+num(r.x,2)+'</td>':'')+'<td>'+esc(metric(r.value,chart.unit))+'</td><td>'+esc(chart.type==='scatter'?num(r.n,0)+' ratings':chart.unit==='USD'?r.n+' / '+r.total+' priced':num(r.n,0)+(chart.unit==='%'?' review mentions':' '+chart.unit))+'</td></tr>';}).join('')+'</tbody></table></div></details>';
  }
  function download(blob,name) {
    var url=URL.createObjectURL(blob),a=document.createElement('a');a.href=url;a.download=name;document.body.appendChild(a);a.click();a.remove();setTimeout(function(){URL.revokeObjectURL(url);},1000);
  }
  function csv(chart) {
    function cell(value) {value=String(value==null?'':value);if(/^[\s]*[=+\-@]|^[\t\r\n]/.test(value))value="'"+value;return '"'+value.replace(/"/g,'""')+'"';}
    var columns=['label','value','x','n','total','coverage','low','high','n_products','reliable'];
    var lines=[['Chart',chart.title],['Scope',chart.scope],['Units',chart.unit],['X axis',chart.x_label],['Y axis',chart.y_label],['Source','Amazon Reviews 2023'],['Notes',(chart.notes||[]).join(' ')],columns];
    chart.rows.forEach(function(r){lines.push(columns.map(function(c){return r[c];}));});
    download(new Blob(['\uFEFF'+lines.map(function(row){return row.map(cell).join(',');}).join('\r\n')],{type:'text/csv;charset=utf-8'}),'laptop-analysis.csv');
  }
  function mountOne(host,chart,expanded) {
    var id='assistant-chart-'+(++serial);
    host.className='chat-chart';
    host.innerHTML='<div class="chat-chart__head"><div><p class="section-kicker">CALCULATED FROM THE DATASET</p><h3>'+esc(chart.title)+'</h3></div></div><p class="chat-chart__scope">'+esc(chart.scope)+'</p><div class="chat-chart__actions">'+(!expanded?'<button type="button" data-chart-expand>Open larger ↗</button>':'')+'<button type="button" data-chart-csv>Download CSV</button><button type="button" data-chart-svg>Save SVG</button></div><div class="chat-chart__plot"></div><p class="chat-chart__axis">'+esc(chart.type==='scatter'?chart.x_label+' · '+chart.y_label:chart.x_label)+'</p><p class="chat-chart__tooltip" role="status">Tap a bar, slice or point to inspect its evidence.</p>'+table(chart)+'<details class="chat-chart__context"><summary>Source &amp; limitations</summary><p>Source: '+esc(chart.source)+' · '+num(chart.n_total,0)+' matching catalogue listings.</p><ul>'+(chart.notes||[]).map(function(note){return '<li>'+esc(note)+'</li>';}).join('')+'</ul></details>';
    var plot=host.querySelector('.chat-chart__plot');
    var lastWidth=0;
    function draw(){var width=Math.max(260,Math.round(host.clientWidth-26));if(width===lastWidth)return;lastWidth=width;plot.innerHTML=svg(chart,width,id);}
    draw();
    if(window.ResizeObserver){var observer=new ResizeObserver(draw);observer.observe(host);host._chartObserver=observer;}
    function inspect(event){var point=event.target.closest('[data-point]');if(!point)return;var row=chart.rows[Number(point.dataset.point)];if(row)host.querySelector('.chat-chart__tooltip').textContent=description(chart,row);}
    plot.addEventListener('click',inspect);plot.addEventListener('mouseover',inspect);plot.addEventListener('focusin',inspect);
    plot.addEventListener('keydown',function(event){if(event.key==='Enter'||event.key===' '){event.preventDefault();inspect(event);}});
    host.querySelector('[data-chart-csv]').addEventListener('click',function(){csv(chart);});
    host.querySelector('[data-chart-svg]').addEventListener('click',function(){
      var node=plot.querySelector('svg').cloneNode(true),styles=getComputedStyle(host);
      node.querySelectorAll('[fill],[stroke]').forEach(function(el){['fill','stroke'].forEach(function(attr){var val=el.getAttribute(attr),match=val&&val.match(/^var\((--[a-z0-9-]+)\)$/);if(match)el.setAttribute(attr,styles.getPropertyValue(match[1]).trim());});});
      var title=document.createElementNS(NS,'title');title.textContent=chart.title+' — '+chart.scope;node.prepend(title);
      var box=node.viewBox.baseVal,w=box.width,h=box.height,group=document.createElementNS(NS,'g');
      group.setAttribute('transform','translate(0 58)');
      Array.from(node.children).forEach(function(child){if(child.tagName!=='title'&&child.tagName!=='desc')group.appendChild(child);});
      node.appendChild(group);node.setAttribute('viewBox','0 0 '+w+' '+(h+98));node.setAttribute('height',h+98);
      var background=document.createElementNS(NS,'rect');background.setAttribute('width',w);background.setAttribute('height',h+98);background.setAttribute('fill',styles.getPropertyValue('--surface-1').trim());node.prepend(background);
      [[chart.title,19,13],[chart.scope,37,10],[chart.x_label,h+75,10],['Amazon Reviews 2023 · historical research dataset',h+91,9]].forEach(function(line){
        var text=document.createElementNS(NS,'text');text.setAttribute('x',0);text.setAttribute('y',line[1]);text.setAttribute('font-size',line[2]);text.setAttribute('fill',styles.getPropertyValue('--ink').trim());text.textContent=cut(line[0],Math.floor(w/(line[2]*.56)));node.appendChild(text);
      });
      download(new Blob([new XMLSerializer().serializeToString(node)],{type:'image/svg+xml;charset=utf-8'}),'laptop-analysis.svg');
    });
    var expand=host.querySelector('[data-chart-expand]');
    if(expand)expand.addEventListener('click',function(){
      var dialog=document.createElement('dialog');dialog.className='chat-chart-dialog';dialog.setAttribute('aria-label',chart.title);
      dialog.innerHTML='<div class="chat-chart-dialog__top"><span>Explore the chart</span><button type="button" aria-label="Close chart" autofocus>×</button></div><div data-expanded-chart></div>';
      document.body.appendChild(dialog);dialog.showModal();
      var inner=dialog.querySelector('[data-expanded-chart]');mountOne(inner,chart,true);
      dialog.querySelector('button').addEventListener('click',function(){dialog.close();});
      dialog.addEventListener('close',function(){if(inner._chartObserver)inner._chartObserver.disconnect();dialog.remove();expand.focus();},{once:true});
      dialog.addEventListener('click',function(event){if(event.target===dialog){var r=dialog.getBoundingClientRect();if(event.clientX<r.left||event.clientX>r.right||event.clientY<r.top||event.clientY>r.bottom)dialog.close();}});
    });
  }
  window.ChatCharts={mount:function(container,charts){(charts||[]).slice(0,3).forEach(function(chart){if(!valid(chart))return;var host=document.createElement('section');container.appendChild(host);mountOne(host,chart,false);});}};
})();
