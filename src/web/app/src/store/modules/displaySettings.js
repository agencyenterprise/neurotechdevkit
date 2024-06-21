export default {
  namespaced: true,
  state() {
    return {
      sliceAxis: "",
      slicePosition: null,
    };
  },
  mutations: {
    setSliceAxis(state, payload) {
      state.sliceAxis = payload;
    },
    setSlicePosition(state, payload) {
      state.slicePosition = payload;
    },
  },
  actions: {
    setDisplaySettings(context, payload) {
      if (payload === null) {
        context.commit("setSliceAxis", "");
        context.commit("setSlicePosition", null);
        return;
      }
      context.commit("setSliceAxis", payload.sliceAxis);
      context.commit("setSlicePosition", payload.slicePosition);
    },
    setSliceAxis(context, payload) {
      context.commit("setSliceAxis", payload);
    },
    setSlicePosition(context, payload) {
      context.commit("setSlicePosition", payload);
    },
  },
  getters: {
    sliceAxis(state) {
      return state.sliceAxis;
    },
    slicePosition(state) {
      return state.slicePosition;
    },
    displaySettings(state) {
      if (state.sliceAxis === "" || state.slicePosition === null) {
        return null;
      }
      return {
        sliceAxis: state.sliceAxis,
        slicePosition: state.slicePosition,
      };
    },
  },
};
