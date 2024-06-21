export default {
  namespaced: true,
  state() {
    return {
      centerX: null,
      centerY: null,
      centerZ: null,
      radius: null,
    };
  },
  mutations: {
    updateTargetProperty(state, payload) {
      state[payload.key] = payload.value;
    },
    setTarget(state, payload) {
      state.centerX = payload.centerX;
      state.centerY = payload.centerY;
      state.centerZ = payload.centerZ;
      state.radius = payload.radius;
    },
  },
  actions: {
    updateTargetProperty({ commit }, payload) {
      commit("updateTargetProperty", payload);
    },
    setTarget({ commit }, payload) {
      commit("setTarget", payload);
    },
  },
  getters: {
    centerX(state) {
      return state.centerX;
    },
    centerY(state) {
      return state.centerY;
    },
    centerZ(state) {
      return state.centerZ;
    },
    radius(state) {
      return state.radius;
    },
    target(state) {
      return {
        centerX: state.centerX,
        centerY: state.centerY,
        centerZ: state.centerZ,
        radius: state.radius,
      };
    },
  },
};
