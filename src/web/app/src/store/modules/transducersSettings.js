export default {
  namespaced: true,
  state() {
    return {
      transducers: [],
    };
  },
  mutations: {
    setTransducers(state, payload) {
      state.transducers = payload;
    },
    deleteTransducer(state, index) {
      state.transducers.splice(index, 1);
    },
    updateTransducer(state, { index, settings }) {
      state.transducers[index] = { ...state.transducers[index], ...settings };
    },
    addTransducer(state, transducerSettings) {
      state.transducers.push(transducerSettings);
    },
  },
  actions: {
    setTransducers(context, payload) {
      context.commit("setTransducers", payload);
    },
    addTransducer({ commit }, transducerSettings) {
      commit("addTransducer", transducerSettings);
    },
  },
  getters: {
    transducers(state) {
      return state.transducers;
    },
  },
};
