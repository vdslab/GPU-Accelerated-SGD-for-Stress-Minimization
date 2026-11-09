// Browser counterpart of tools/gpu_visualizer: GPU edges + instanced circular nodes.
struct Uniforms {
  transform: vec4<f32>, // scale, offset x/y, node radius in logical pixels
  style: vec4<f32>,     // edge width, maximum BFS distance, edge opacity, logical pixel size
  counts: vec4<u32>,    // vertices, edge stride, first local edge, edges in buffer
}
@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage,read> positions: array<vec2<f32>>;
@group(0) @binding(2) var<storage,read> edges: array<vec2<u32>>;
@group(0) @binding(3) var<storage,read> distances: array<u32>;

fn logical(p:vec2<f32>) -> vec2<f32> {return p*u.transform.x + u.transform.yz;}
fn clip(p:vec2<f32>) -> vec4<f32> {
  return vec4<f32>(p.x/512.0-1.0, 1.0-p.y/512.0,0.0,1.0);
}
fn corner(vertex:u32)->vec2<f32> {
  var corners=array<vec2<f32>,6>(vec2(-1.0,-1.0),vec2(1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,-1.0),vec2(1.0,1.0),vec2(-1.0,1.0));
  return corners[vertex];
}
struct EdgeOut {
  @builtin(position) position:vec4<f32>,
  @location(0) side:f32,
}
@vertex fn edge_vertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) instance:u32)->EdgeOut {
  let pair=edges[u.counts.z+instance*u.counts.y];
  let a=logical(positions[pair.x]); let b=logical(positions[pair.y]);
  let difference=b-a; let direction=difference/max(length(difference),1e-8);
  let c=corner(vertex);
  let half_width=u.style.x*0.5+u.style.w;
  let normal=vec2<f32>(-direction.y,direction.x)*half_width;
  return EdgeOut(clip(mix(a,b,(c.x+1.0)*0.5)+normal*c.y),c.y*half_width);
}
@fragment fn edge_fragment(in:EdgeOut)->@location(0) vec4<f32> {
  let aa=max(fwidth(in.side),0.001);
  let coverage=clamp((u.style.x*0.5-abs(in.side))/aa+0.5,0.0,1.0);
  return vec4<f32>(vec3<f32>(100.0,116.0,139.0)/255.0,u.style.z*coverage);
}
struct NodeOut {
  @builtin(position) position:vec4<f32>,
  @location(0) local:vec2<f32>,
  @location(1) color:vec3<f32>,
}
@vertex fn node_vertex(@builtin(vertex_index) vertex:u32,@builtin(instance_index) node:u32)->NodeOut {
  let c=corner(vertex);
  let t=f32(min(31u,u32(floor(f32(distances[node])/max(1.0,u.style.y)*31.0))))/31.0*2.0;
  let purple=vec3<f32>(86.0,42.0,129.0);
  let red=vec3<f32>(220.0,76.0,60.0);
  let yellow=vec3<f32>(252.0,231.0,149.0);
  var color=mix(purple,red,min(1.0,t));
  if(t>1.0) {color=mix(red,yellow,t-1.0);}
  let local=c*(u.transform.w+u.style.w);
  return NodeOut(clip(logical(positions[node])+local),local,round(color)/255.0);
}
@fragment fn node_fragment(in:NodeOut)->@location(0) vec4<f32> {
  let radius=length(in.local); let aa=max(fwidth(radius),0.001);
  let coverage=clamp((u.transform.w-radius)/aa+0.5,0.0,1.0);
  if(coverage<=0.0){discard;}
  return vec4<f32>(in.color,coverage);
}
