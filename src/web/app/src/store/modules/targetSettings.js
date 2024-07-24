const initialState = () => ({
  centerX: "0.14399999141693112",
  centerY: "0.14399999141693112",
  centerZ: "0.14399999136",
  radius: "0.01",
});
export default {
  namespaced: true,
  state: initialState,
  mutations: {
    reset(state) {
      Object.assign(state, initialState());
    },
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
    targetPayload(state, getters, rootState, { is2d }) {
      return {
        centerX: state.centerX,
        centerY: state.centerY,
        centerZ: is2d ? null : state.centerZ,
        radius: state.radius,
      };
    },
  },
};
