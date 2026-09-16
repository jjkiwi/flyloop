// Forward kinematics for the exported rig, transcribed from
// FlyRig.forward_kinematics in flyloop/viz/fly_geometry.py.
//
// The order matters and is the whole reason that function has a test against
// MuJoCo's mj_kinematics: a body is placed in its parent's frame, then each of
// its joints is applied in document order, each axis interpreted in the frame
// the joints before it left. NeuroMechFly puts up to three hinges on a coxa,
// so composing them in the wrong order is wrong by tens of degrees at the
// tarsus. The anchor correction (rotate about joint.pos, not the body origin)
// is a no-op for this model -- every joint anchors at zero -- and is kept
// because dropping it would make the reader silently wrong for the next MJCF.
//
// Quaternions in rig.json are already Three.js order (x, y, z, w); rig.json
// says so in "quaternion_order", and this file asserts it.

export function quatMul(a, b, out) {
  const ax = a[0], ay = a[1], az = a[2], aw = a[3];
  const bx = b[0], by = b[1], bz = b[2], bw = b[3];
  out[0] = aw * bx + ax * bw + ay * bz - az * by;
  out[1] = aw * by - ax * bz + ay * bw + az * bx;
  out[2] = aw * bz + ax * by - ay * bx + az * bw;
  out[3] = aw * bw - ax * bx - ay * by - az * bz;
  return out;
}

export function quatRotate(q, v, out) {
  const qx = q[0], qy = q[1], qz = q[2], qw = q[3];
  // t = 2 * (q_vec x v); out = v + qw * t + q_vec x t
  const tx = 2 * (qy * v[2] - qz * v[1]);
  const ty = 2 * (qz * v[0] - qx * v[2]);
  const tz = 2 * (qx * v[1] - qy * v[0]);
  out[0] = v[0] + qw * tx + qy * tz - qz * ty;
  out[1] = v[1] + qw * ty + qz * tx - qx * tz;
  out[2] = v[2] + qw * tz + qx * ty - qy * tx;
  return out;
}

function axisAngleQuat(axis, angle, out) {
  const n = Math.hypot(axis[0], axis[1], axis[2]) || 1;
  const s = Math.sin(0.5 * angle) / n;
  out[0] = axis[0] * s; out[1] = axis[1] * s; out[2] = axis[2] * s;
  out[3] = Math.cos(0.5 * angle);
  return out;
}

export class Rig {
  constructor(rig) {
    if (rig.quaternion_order !== "xyzw") {
      throw new Error(`rig.json says quaternion_order=${rig.quaternion_order}; this reader wants xyzw`);
    }
    this.raw = rig;
    this.bodies = rig.bodies;
    this.joints = rig.joints;
    this.nBodies = rig.bodies.length;
    this.nJoints = rig.joints.length;

    // joints[:, i] of the episode is named by rig.joint_order[i]; the rig's own
    // joint list is longer (87 hinges against 42 logged degrees of freedom --
    // head, abdomen, wings and the distal tarsi are not actuated), so the rest
    // stay at zero.
    const byName = new Map(rig.joints.map((j, i) => [j.name, i]));
    this.slots = (rig.joint_order || []).map((name) => {
      if (!byName.has(name)) throw new Error(`joint_order names ${name}, absent from rig.joints`);
      return byName.get(name);
    });

    this.angles = new Float64Array(this.nJoints);
    this.pos = new Float64Array(this.nBodies * 3);
    this.quat = new Float64Array(this.nBodies * 4);
    this._p = new Float64Array(3);
    this._q = new Float64Array(4);
    this._t = new Float64Array(3);
    this._u = new Float64Array(4);
  }

  /** Load one logged row (length joint_order.length) into the angle vector. */
  setLoggedAngles(row) {
    this.angles.fill(0);
    for (let i = 0; i < this.slots.length; i++) {
      const v = row[i];
      this.angles[this.slots[i]] = Number.isFinite(v) ? v : 0;
    }
  }

  /** World pose of every body, in the root body's frame. */
  solve() {
    const { bodies, joints, angles, pos, quat } = this;
    const p = this._p, q = this._q, t = this._t, u = this._u;
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      if (b.parent < 0) {
        p[0] = p[1] = p[2] = 0;
        q[0] = q[1] = q[2] = 0; q[3] = 1;
      } else {
        const pi = b.parent;
        p[0] = pos[pi * 3]; p[1] = pos[pi * 3 + 1]; p[2] = pos[pi * 3 + 2];
        q[0] = quat[pi * 4]; q[1] = quat[pi * 4 + 1]; q[2] = quat[pi * 4 + 2]; q[3] = quat[pi * 4 + 3];
      }
      quatRotate(q, b.pos, t);
      p[0] += t[0]; p[1] += t[1]; p[2] += t[2];
      quatMul(q, b.quat, u);
      q[0] = u[0]; q[1] = u[1]; q[2] = u[2]; q[3] = u[3];

      for (const ji of b.joints) {
        const j = joints[ji];
        quatRotate(q, j.pos, t);                       // anchor, in world
        const ax = p[0] + t[0], ay = p[1] + t[1], az = p[2] + t[2];
        axisAngleQuat(j.axis, angles[ji], u);
        quatMul(q, u, u);
        q[0] = u[0]; q[1] = u[1]; q[2] = u[2]; q[3] = u[3];
        quatRotate(q, j.pos, t);
        p[0] = ax - t[0]; p[1] = ay - t[1]; p[2] = az - t[2];
      }

      pos[i * 3] = p[0]; pos[i * 3 + 1] = p[1]; pos[i * 3 + 2] = p[2];
      quat[i * 4] = q[0]; quat[i * 4 + 1] = q[1]; quat[i * 4 + 2] = q[2]; quat[i * 4 + 3] = q[3];
    }
    return this;
  }
}
